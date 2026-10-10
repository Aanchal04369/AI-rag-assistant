import React, { useState } from "react";
import axios from "axios";
import {
  BrainCircuit, FlaskConical, BarChart3,
  Loader2, CheckCircle2, AlertCircle, ChevronDown, ChevronUp
} from "lucide-react";

export default function NLPPanel({ apiUrl }) {
  const [finetuneEpochs, setFinetuneEpochs] = useState(1);
  const [finetuneLoading, setFinetuneLoading] = useState(false);
  const [finetuneResult, setFinetuneResult] = useState(null);
  const [finetuneError, setFinetuneError] = useState("");

  const [evalLoading, setEvalLoading] = useState(false);
  const [evalResult, setEvalResult] = useState(null);
  const [evalError, setEvalError] = useState("");

  const [pipelineLoading, setPipelineLoading] = useState(false);
  const [pipelineResult, setPipelineResult] = useState(null);
  const [pipelineOpen, setPipelineOpen] = useState(false);

  // ── Fine-tune ──────────────────────────────────────────────────────
  const runFinetune = async () => {
    setFinetuneLoading(true);
    setFinetuneError("");
    setFinetuneResult(null);
    try {
      const res = await axios.post(`${apiUrl}/finetune`, { epochs: finetuneEpochs });
      setFinetuneResult(res.data);
    } catch (err) {
      setFinetuneError(err.response?.data?.error || "Fine-tuning failed.");
    } finally {
      setFinetuneLoading(false);
    }
  };

  // ── Evaluate ──────────────────────────────────────────────────────
  const runEvaluate = async () => {
    setEvalLoading(true);
    setEvalError("");
    setEvalResult(null);
    try {
      const res = await axios.post(`${apiUrl}/evaluate`);
      setEvalResult(res.data);
    } catch (err) {
      setEvalError(err.response?.data?.error || "Evaluation failed.");
    } finally {
      setEvalLoading(false);
    }
  };

  // ── Pipeline status ───────────────────────────────────────────────
  const loadPipeline = async () => {
    if (pipelineOpen) { setPipelineOpen(false); return; }
    setPipelineLoading(true);
    try {
      const res = await axios.get(`${apiUrl}/pipeline-status`);
      setPipelineResult(res.data);
      setPipelineOpen(true);
    } catch (err) {
      setPipelineResult({ error: err.response?.data?.error || "Could not load pipeline stats." });
      setPipelineOpen(true);
    } finally {
      setPipelineLoading(false);
    }
  };

  return (
    <div className="nlp-panel">
      <div className="nlp-section-title">
        <BrainCircuit size={15} /> NLP + Fine-Tuning
      </div>

      {/* Pipeline Status */}
      <button className="nlp-btn nlp-btn-outline" onClick={loadPipeline} disabled={pipelineLoading}>
        {pipelineLoading
          ? <><Loader2 className="spin" size={13} /> Loading…</>
          : <>{pipelineOpen ? <ChevronUp size={13} /> : <ChevronDown size={13} />} NLP Pipeline Stats</>
        }
      </button>

      {pipelineOpen && pipelineResult && (
        <div className="nlp-result">
          {pipelineResult.error
            ? <span className="nlp-error">{pipelineResult.error}</span>
            : <>
                <div className="nlp-stat"><b>Sentences</b> {pipelineResult.pipeline_stats?.sentence_count ?? "—"}</div>
                <div className="nlp-stat"><b>Tokens</b> {pipelineResult.pipeline_stats?.token_count ?? "—"}</div>
                <div className="nlp-stat"><b>Stages done</b> {(pipelineResult.pipeline_stats?.stages_completed || []).length > 0 ? (pipelineResult.pipeline_stats.stages_completed).join(" → ") : "Re-upload PDF to run pipeline"}</div>
              </>
          }
        </div>
      )}

      {/* Fine-Tune */}
      <div className="nlp-row">
        <label className="nlp-label">Epochs (1–20)</label>
        <input
          type="number" min={1} max={20} value={finetuneEpochs}
          onChange={e => setFinetuneEpochs(Number(e.target.value))}
          className="nlp-input"
        />
      </div>

      <button className="nlp-btn nlp-btn-primary" onClick={runFinetune} disabled={finetuneLoading || evalLoading}>
        {finetuneLoading
          ? <><Loader2 className="spin" size={13} /> Fine-tuning BERT…</>
          : <><FlaskConical size={13} /> Fine-tune DistilBERT</>
        }
      </button>

      {finetuneError && <div className="nlp-error">{finetuneError}</div>}

      {finetuneResult && (
        <div className="nlp-result">
          <CheckCircle2 size={13} className="nlp-ok" /> Fine-tuning complete
          <div className="nlp-stat"><b>Examples</b> {finetuneResult.examples}</div>
          <div className="nlp-stat"><b>Model saved to</b> {finetuneResult.model_dir}</div>
          {finetuneResult.epoch_logs?.map(e => (
            <div className="nlp-stat" key={e.epoch}>
              Epoch {e.epoch} — loss: {e.loss}
            </div>
          ))}
        </div>
      )}

      {/* Evaluate */}
      <button className="nlp-btn nlp-btn-secondary" onClick={runEvaluate} disabled={evalLoading || finetuneLoading}>
        {evalLoading
          ? <><Loader2 className="spin" size={13} /> Evaluating…</>
          : <><BarChart3 size={13} /> Evaluate (BERTScore + F1)</>
        }
      </button>

      {evalError && <div className="nlp-error">{evalError}</div>}

      {evalResult && (
        <div className="nlp-result">
          {evalResult.warning && (
            <div className="nlp-warn"><AlertCircle size={12} /> {evalResult.warning}</div>
          )}
          <div className="nlp-metrics">
            <div className="nlp-metric-box">
              <span>BERTScore F1</span>
              <strong>{(evalResult.bertscore_f1 * 100).toFixed(1)}%</strong>
            </div>
            <div className="nlp-metric-box">
              <span>Token F1</span>
              <strong>{(evalResult.token_f1 * 100).toFixed(1)}%</strong>
            </div>
            <div className="nlp-metric-box">
              <span>Examples</span>
              <strong>{evalResult.example_count}</strong>
            </div>
          </div>
          <div className="nlp-stat" style={{ fontSize: "9px", color: "#888" }}>
            Precision {(evalResult.bertscore_precision * 100).toFixed(1)}% · Recall {(evalResult.bertscore_recall * 100).toFixed(1)}%
          </div>
        </div>
      )}
    </div>
  );
}
