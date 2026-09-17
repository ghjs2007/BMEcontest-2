import { useEffect, useState } from 'react';
import type { Prediction } from '../data/types';

type Release = { run_key: string; f1: number; outer_metrics?: { f1: number; ppv: number; sensitivity: number; n_tp: number; n_true: number; n_pred: number } };

/** Promoted-release evidence, loaded from generated runtime metadata (single source of truth). */
export default function ModelPage({ prediction }: { prediction: Prediction }) {
  const [release, setRelease] = useState<Release | null>(null);
  useEffect(() => {
    fetch('./runtime/release-metadata.json').then(r => r.ok ? r.json() : null).then(setRelease).catch(() => {});
  }, []);
  const m = release?.outer_metrics;
  return (
    <section className="panel model-page">
      <div className="page-title"><div><h1>Model</h1><p>Promoted release metadata and inference contract.</p></div></div>
      <div className="model-grid">
        <div>
          <h2>Release evidence</h2>
          <p>Subject-disjoint five-fold cross-validation development evidence. Not an independent test-set score.</p>
          <dl>
            <dt>Run key</dt><dd className="mono">{release?.run_key || prediction.model.run_key}</dd>
            <dt>F1</dt><dd>{m?.f1?.toFixed(3) || release?.f1?.toFixed(3) || '—'}</dd>
            <dt>Precision / PPV</dt><dd>{m?.ppv?.toFixed(3) || '—'}</dd>
            <dt>Recall / sensitivity</dt><dd>{m?.sensitivity?.toFixed(3) || '—'}</dd>
            <dt>TP / eligible / predictions</dt><dd>{m ? `${m.n_tp} / ${m.n_true} / ${m.n_pred}` : '—'}</dd>
          </dl>
        </div>
        <div>
          <h2>Inference path</h2>
          <div className="pipeline">Raw wrist IMU <span>↓</span> Macro / Micro <span>↓</span> Candidate admission <span>↓</span> Context + event verifier <span>↓</span> Eating episodes</div>
          <p className="muted">Visualization only consumes the published prediction contract and optional motion telemetry. Canonical inference always runs in Python.</p>
        </div>
      </div>
    </section>
  );
}
