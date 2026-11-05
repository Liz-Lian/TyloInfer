#pragma once

static void check_resident_depth_policy() {
    using mfq::engine::mtp::DepthController;
    const auto elapsed=[](int depth){return depth?18.0+1.5*depth:12.0;};
    int cases=0;
    for(bool blockwise:{false,true})for(int maximum:{1,3,7})for(bool periodic:{false,true}) {
        DepthController policy(maximum,nullptr,7.0,1.03,true,0.0,blockwise,periodic);
        int rounds=0,ordinary=0;
        while(policy.warming_up() && rounds++<160) {
            const int depth=policy.depth();
            const auto previous=policy.measured_cycle_ms(depth);
            // Graph construction and maintenance must not consume a
            // calibration slot or replace a measured execution cost.
            policy.observe(depth,depth,1000.0,false);
            require(policy.depth()==depth && policy.measured_cycle_ms(depth)==previous);
            ordinary+=depth==0;
            policy.observe(depth,depth,elapsed(depth));
        }
        require(rounds<160 && !policy.warming_up());
        require(ordinary==(periodic?8:2));
        for(int depth=0;depth<=maximum;++depth)require(policy.measured_cycle_ms(depth).has_value());
        ++cases;
    }
    for(bool blockwise:{false,true}) {
        DepthController seed(7,nullptr,7.0,1.03,true,0.0,blockwise,false);
        for(int step=0;step<100;++step) {
            const int depth=seed.depth();seed.observe(depth,depth,elapsed(depth));
        }
        require(!seed.warming_up());
        DepthController resumed(7,&seed,7.0,1.03,true,0.0,blockwise,false);
        const int preferred=resumed.depth();
        require(preferred>0 && seed.measured_cycle_ms(preferred));
        int rounds=0,ordinary=0;
        while(resumed.warming_up() && rounds++<32) {
            const int depth=resumed.depth();ordinary+=depth==0;
            require(depth==0 || depth==preferred);
            resumed.observe(depth,depth,elapsed(depth));
        }
        require(rounds==4 && ordinary==2);
        // A changed ordinary cost invalidates relative width costs. This
        // models a new context or placement cost, without device fixtures.
        DepthController changed(7,&seed,7.0,1.03,true,0.0,blockwise,false);
        int measured=0;
        for(int step=0;step<80;++step) {
            const int depth=changed.depth();
            changed.observe(depth,depth,elapsed(depth)+12.0);
        }
        for(int depth=2;depth<=7;++depth) {
            const auto cost=changed.measured_cycle_ms(depth);
            require(cost && *cost>=elapsed(depth)+11.9);++measured;
        }
        require(measured==6);
        // A larger request cap must discover widths absent from the seed;
        // a smaller cap must keep all selected widths within its new limit.
        DepthController small(2,nullptr,7.0,1.03,true,0.0,blockwise,false);
        for(int step=0;step<50;++step){const int d=small.depth();small.observe(d,d,elapsed(d));}
        DepthController larger(7,&small,7.0,1.03,true,0.0,blockwise,false);
        for(int step=0;step<64;++step){const int d=larger.depth();require(d<=7);larger.observe(d,d,elapsed(d));}
        for(int depth=3;depth<=7;++depth)require(larger.measured_cycle_ms(depth).has_value());
        DepthController smaller(3,&seed,7.0,1.03,true,0.0,blockwise,false);
        for(int step=0;step<40;++step){const int d=smaller.depth();require(d<=3);smaller.observe(d,d,elapsed(d));}
        // Warm acceptance priors must still yield to low-acceptance text.
        DepthController losing(7,&seed,7.0,1.03,true,0.0,blockwise,false);
        bool exit=false;
        for(int step=0;step<160;++step) {
            const int d=losing.depth();losing.observe(d,0,elapsed(d));
            if(losing.should_exit()){exit=true;break;}
        }
        require(exit);cases+=5;
    }
    for(bool blockwise:{false,true})for(bool periodic:{false,true}) {
        DepthController policy(3,nullptr,7.0,1.03,true,0.0,blockwise,periodic);
        for(int step=0;policy.warming_up() && step<160;++step) {
            const int depth=policy.depth();policy.observe(depth,depth,elapsed(depth));
        }
        require(!policy.warming_up() && policy.depth()>0);
        // A long response refreshes ordinary cost. Resident weights need
        // one execution; streamed weights still sample an exchange period.
        for(int step=0;policy.depth()>0 && step<300;++step) {
            const int depth=policy.depth();policy.observe(depth,depth,elapsed(depth));
        }
        require(policy.depth()==0);
        policy.observe(0,0,1000.0,false);
        require(policy.depth()==0);
        for(int sample=0;sample<(periodic?4:1);++sample) {
            require(policy.depth()==0);
            policy.observe(0,0,elapsed(0));
        }
        require(policy.depth()>0);++cases;
    }
    std::cout<<"Resident/streamed MTP calibration, warm seeds, cost drift, caps and acceptance cases="
        <<cases<<" PASS\n";
}
