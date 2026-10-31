#pragma once
// Cost fits and PCIe-count search adapted from Strata's PcieModel (MIT;
// ../../../../third_party/strata.LICENSE). Times are measured on the running
// machine; no device identity or benchmark dimensions enter the decision.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

struct GgufCpuBalanceModel {
    bool on = false;
    double a = 0, c = 0, d = 0, g0 = 0, g = 0;   ///< the fits (read by `pick`)
    double A[3][3] = {}, B[3] = {};   ///< the CPU fit's normal equations, [1, n_c, n_c+n_p]
    double C[2][2] = {}, D[2] = {};   ///< the GPU fit's, [1, n_p]
    int ncpu = 0, ngpu = 0, ngpu_p = 0;   ///< samples in each (ngpu_p: with a PCIe expert)
    uint32_t explore = 0;
    std::uint64_t deferred_probes=0;
    std::uint64_t risk_recovery_probes=0;
    double tail_variance = 0;
    double symmetric_variance = 0;   ///< diagnostic comparison with the former two-sided estimate
    /// One layer's times: the pool's `p` with `n_c` CPU and `n_p` PCIe experts, the GPU's `gpu` (ms).
    void sample(int n_c, int n_p, double p, double gpu) {
        constexpr double lam = 0.995;   // ~200 layers' memory: a few windows
        if (n_c > 0) {
            // p = a + (c-d)*CPU + d*(CPU+PCIe). All three coefficients
            // are nonnegative; this also represents the constraint d <= c.
            const double x[3] = {1.0, (double) n_c, double(n_c+n_p)};
            for (int i = 0; i < 3; ++i) {
                B[i] = lam * B[i] + x[i] * p;
                for (int j = 0; j < 3; ++j) A[i][j] = lam * A[i][j] + x[i] * x[j];
            }
            ++ncpu;
        }
        const double y[2] = {1.0, (double) n_p};
        for (int i = 0; i < 2; ++i) {
            D[i] = lam * D[i] + y[i] * gpu;
            for (int j = 0; j < 2; ++j) C[i][j] = lam * C[i][j] + y[i] * y[j];
        }
        ++ngpu;
        ngpu_p += n_p > 0;
    }
    /// Solve both fits (after a window). A ridge keeps a fit whose counts barely vary near its mean ratio.
    void solve() {
        if (ncpu >= 8) {
            // Independent post-fit clamping biases predictions when CPU+PCIe
            // is nearly constant. Solve the three-variable nonnegative ridge
            // least squares exactly by enumerating its eight active sets.
            double best=0,values[3]={};
            const double ridge=1e-3*A[0][0];
            for(unsigned mask=1;mask<8;++mask) {
                int active[3],n=0;
                for(int i=0;i<3;++i)if(mask&(1u<<i))active[n++]=i;
                double M[3][4]={};
                for(int i=0;i<n;++i) {
                    for(int j=0;j<n;++j)M[i][j]=A[active[i]][active[j]]+
                        (active[i]==active[j] && active[i]>0?ridge:0.);
                    M[i][n]=B[active[i]];
                }
                bool valid=true;
                for(int k=0;k<n && valid;++k) {
                    int pivot=k;
                    for(int i=k+1;i<n;++i)if(std::abs(M[i][k])>std::abs(M[pivot][k]))pivot=i;
                    if(std::abs(M[pivot][k])<1e-12){valid=false;break;}
                    for(int j=0;j<=n;++j)std::swap(M[k][j],M[pivot][j]);
                    for(int i=0;i<n;++i)if(i!=k) {
                        const double scale=M[i][k]/M[k][k];
                        for(int j=k;j<=n;++j)M[i][j]-=scale*M[k][j];
                    }
                }
                double beta[3]={};
                for(int i=0;i<n && valid;++i) {
                    const double value=M[i][n]/M[i][i];
                    if(!std::isfinite(value) || value < -1e-9)valid=false;
                    else beta[active[i]]=std::max(value,0.);
                }
                if(!valid)continue;
                double loss=0;
                for(int i=0;i<3;++i) {
                    loss-=2*B[i]*beta[i];
                    for(int j=0;j<3;++j)loss+=beta[i]*A[i][j]*beta[j];
                    if(i>0)loss+=ridge*beta[i]*beta[i];
                }
                if(loss<best-1e-12){best=loss;std::copy_n(beta,3,values);}
            }
            a=values[0];d=values[2];c=values[1]+d;
        }
        if (ngpu >= 8 && ngpu_p >= 4) {
            // Solve the nonnegative two-variable fit, including its boundary
            // solutions. Clamping a negative OLS intercept while keeping its
            // slope overprices every GPU split, favoring unnecessary CPU work.
            double best=0,intercept=0,slope=0;
            const auto consider=[&](double a,double b) {
                if(a<0 || b<0 || !std::isfinite(a) || !std::isfinite(b))return;
                const auto loss=a*a*C[0][0]+2*a*b*C[0][1]+b*b*C[1][1]-2*a*D[0]-2*b*D[1];
                if(loss<best){best=loss;intercept=a;slope=b;}
            };
            if(C[0][0]>0)consider(std::max(0.,D[0]/C[0][0]),0.);
            if(C[1][1]>0)consider(0.,std::max(0.,D[1]/C[1][1]));
            const double det = C[0][0] * C[1][1] - C[0][1] * C[1][0];
            if (std::abs(det) > 1e-12) {
                consider((D[0]*C[1][1]-C[0][1]*D[1])/det,
                    (C[0][0]*D[1]-C[1][0]*D[0])/det);
            }
            g0=intercept;g=slope;
        }
    }
    bool ready() const { return on && ncpu >= 8 && ngpu_p >= 4; }
    double mean_price(int misses,int pcie) const {
        const double gpu=g0+g*pcie;
        return misses==pcie?gpu:std::max(gpu,a+c*(misses-pcie)+d*pcie);
    }
    /// Price the whole layer, including the CPU tail when the two lanes jitter.
    double price(int misses,int pcie) const {
        const double cpu=a+c*(misses-pcie)+d*pcie,gpu=g0+g*pcie;
        const double delta=gpu-cpu;
        return misses==pcie?gpu:.5*(gpu+cpu+
            std::sqrt(delta*delta+(2./3.14159265358979323846)*tail_variance));
    }
    /// Keep measuring rejected lanes, with an exploration interval priced by
    /// their observed cost. A millisecond-scale CPU wake must not be retried
    /// every 64 sub-millisecond layers after the fit has already rejected it.
    int pick(int nmiss, int fixed) {
        int m = fixed;
        double best=1e300;
        if (ready()) {
            for (int k = 0; k <= nmiss; ++k) {
                const double t=price(nmiss,k);
                if (t < best - 1e-6) { best = t; m = k; }
            }
        }
        if(on && nmiss>=1) {
            const int probe=m==0?1:m==nmiss?nmiss-1:m;
            uint32_t interval=64;
            bool risk_recovery=false;
            if(ready() && probe!=m && best>0.) {
                const double observed=price(nmiss,probe),fitted=mean_price(nmiss,probe);
                // A risk penalty protects normal work, but uncertainty alone
                // must not postpone the measurement needed to update it.
                // Truly expensive fitted CPU work retains its priced backoff.
                risk_recovery=fitted<best && observed>2*best;
                const double cost=fitted<best?fitted:observed;
                const double weighted=64.*std::max(1.,(cost-best)/best);
                interval=std::isfinite(weighted)?static_cast<uint32_t>(std::min(std::ceil(weighted),
                    double(std::numeric_limits<uint32_t>::max()))):std::numeric_limits<uint32_t>::max();
            }
            ++explore;
            if(interval>64 && explore%64==0)++deferred_probes;
            if(explore>=interval){explore=0;m=probe;if(risk_recovery)++risk_recovery_probes;}
        }
        return m;
    }
    int max_observed_cpu=0;
    void observe(int cpu,int pcie,double cpu_ms,double gpu_ms) {
        if(ready() && cpu>0) {
            // Use the difference residual directly: shared DRAM or scheduling
            // delays can move both lanes together without creating a CPU tail.
            const double residual=(cpu_ms-(a+c*cpu+d*pcie))-(gpu_ms-(g0+g*pcie));
            // Only unexpectedly late CPU completion can expose extra CPU
            // wait. A GPU-only stall (or earlier CPU completion) must not
            // make the CPU lane appear riskier. Twice the positive second
            // moment retains the variance scale for symmetric normal jitter.
            const auto late=std::max(0.,residual);
            tail_variance=.95*tail_variance+.1*late*late;
            symmetric_variance=.95*symmetric_variance+.05*residual*residual;
        }
        max_observed_cpu=std::max(max_observed_cpu,cpu);
        sample(cpu,pcie,cpu_ms,gpu_ms);solve();
    }
    int cpu_count(int missing) {
        if(missing<=0)return 0;
        on=true;
        // Vary CPU counts during calibration so its intercept and per-expert
        // slope are identifiable. A sole miss alternates with GPU-only work.
        int calibration=missing==1?(explore&1u?0:1):(explore&3u?1:2);
        const auto chosen=missing-pick(missing,missing-calibration);
        // A fit trained only on one expert must not extrapolate immediately
        // to an entire layer of CPU work; grow its observed range gradually.
        return std::clamp(chosen,0,std::min(missing,max_observed_cpu+1));
    }
};
