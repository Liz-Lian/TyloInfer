
import { useEffect, useState } from 'react';
import { loadVoiceClip } from '../../realtimeAudio';


export function AudioClip({ audioId }: { audioId: string }) {
  const [url, setUrl] = useState("");
  useEffect(() => {
    let current = true;
    let objectUrl = "";
    loadVoiceClip(audioId)
      .then((blob) => {
        if (!blob || !current) return;
        objectUrl = URL.createObjectURL(blob);
        setUrl(objectUrl);
      })
      .catch(() => undefined);
    return () => {
      current = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [audioId]);
  return url ? <audio className="message-audio" controls preload="metadata" src={url} /> : null;
}
