import React, { useState } from "react";
import axios from "axios";
import { BrainCircuit, Plus, RotateCcw, FileText, ShieldCheck } from "lucide-react";
import FileUpload from "./components/FileUpload";
import QueryInterface from "./components/QueryInterface";
import NLPPanel from "./components/NLPPanel";
import "./App.css";

const API_URL = "http://localhost:5000/api";

export default function App() {
  const [file, setFile] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState("");

  const upload = async (selected) => {
    setProcessing(true); setError("");
    const form = new FormData();
    form.append("file", selected);
    try {
      const res = await axios.post(`${API_URL}/upload`, form);
      setFile(res.data);
    } catch (err) {
      setError(err.response?.data?.error || "Unable to process the PDF.");
    } finally { setProcessing(false); }
  };

  const reset = async () => {
    try { await axios.post(`${API_URL}/reset`); } catch {}
    setFile(null); setError("");
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon"><BrainCircuit size={22}/></div>
          <div>
            <strong>AI Research Assistant</strong>
            <small>Powered by RAG + Gemini AI</small>
          </div>
        </div>

        {file && (
          <button className="new-chat" onClick={reset}>
            <Plus size={16}/> New Chat
          </button>
        )}
      </header>

      {!file ? (
        <main className="landing">
          <div className="hero-icon"><BrainCircuit size={42}/></div>
          <h1>Getting Started</h1>
          <p>Create a research chat by uploading a PDF.</p>

          <button
            className="create-chat"
            onClick={() => document.querySelector(".drop-zone")?.click()}
          >
            <Plus size={18}/> Create New Chat
          </button>

          <FileUpload onUpload={upload} processing={processing}/>

          <div className="privacy">
            <ShieldCheck size={16}/>
            Your PDF is processed locally for retrieval; relevant text is sent to Gemini to generate answers.
          </div>

          {error && <div className="error">{error}</div>}
        </main>
      ) : (
        <main className="workspace">
          <aside className="sidebar">
            <div className="side-title">Current document</div>

            <div className="document-card">
              <FileText size={19}/>
              <div>
                <b>{file.filename}</b>
                <span>{file.chunks_count} searchable chunks</span>
                {file.nlp_pipeline && (
                  <span style={{marginTop:"2px"}}>
                    NLP: {file.nlp_pipeline.token_count?.toLocaleString()} tokens · {file.nlp_pipeline.stages_completed?.length}/5 stages
                  </span>
                )}
              </div>
            </div>

            <button className="sidebar-new" onClick={reset}>
              <RotateCcw size={15}/> Upload another PDF
            </button>

            <NLPPanel apiUrl={API_URL} />

            <div className="side-note">
              Semantic Retrieval + Gemini AI<br/>
              Powered by Google Gemini API.
            </div>
          </aside>

          <section className="chat-area">
            <div className="chat-title">
              <h2>{file.filename}</h2>
              <span>Document Q&A</span>
            </div>

            <QueryInterface
              apiUrl={API_URL}
              filename={file.filename}
            />

            {error && <div className="error">{error}</div>}
          </section>
        </main>
      )}
    </div>
  );
}