
import { useSettings } from '../features/settings/SettingsProvider';
import mfqMark from '../assets/mfq-mark.svg';


export function LoadingPage() {
  const { tr } = useSettings();

  return (
    <section aria-labelledby="loading-title" className="failure-page loading-page" role="status">
      <div className="failure-page-content">
        <div className="failure-page-heading">
          <span className="failure-page-symbol loading-page-symbol" aria-hidden="true" />
        </div>
        <h1 id="loading-title">{tr('正在加载中', 'Loading…')}</h1>
      </div>
      <div className="failure-page-footer" aria-hidden="true">
        <img src={mfqMark} alt="" />
        <span>TyloInfer</span>
      </div>
    </section>
  );
}
