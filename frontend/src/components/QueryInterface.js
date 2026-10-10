import React, { useState } from "react";
import { Send, Sparkles, Search, BookOpen, Loader2 } from "lucide-react";
import axios from "axios";

export default function QueryInterface({ apiUrl, filename }) {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [summaryLoading, setSummaryLoading] = useState(false);

  const ask = async (text = question) => {
    const q = text.trim();
    if (!q || loading || summaryLoading) return;
    const history = messages.map(m => ({ role: m.role, text: m.text }));
    setMessages(m => [...m, { role: "user", text: q }]);
    setQuestion("");
    setLoading(true);
    try {
      const res = await axios.post(`${apiUrl}/query`, { question: q, history });
      setMessages(m => [...m, {
        role: "assistant", text: res.data.answer,
        sources: res.data.relevant_chunks || [], model: res.data.model
      }]);
    } catch (err) {
      setMessages(m => [...m, { role: "assistant", text: err.response?.data?.error || "Something went wrong." }]);
    } finally { setLoading(false); }
  };

  const summarize = async () => {
    if (loading || summaryLoading) return;
    setSummaryLoading(true);
    setMessages(m => [...m, { role: "user", text: "Summarize this research paper." }]);
    try {
      const res = await axios.post(`${apiUrl}/summary`);
      setMessages(m => [...m, { role: "assistant", text: res.data.answer, model: res.data.model, summary: true }]);
    } catch (err) {
      setMessages(m => [...m, { role: "assistant", text: err.response?.data?.error || "Unable to create the summary." }]);
    } finally { setSummaryLoading(false); }
  };

  const factualPrompt = "What are the key findings, results, and conclusions of this paper?";

  return (
    <div className="chat-shell">
      <div className="chat-toolbar">
        <div>
          <div className="paper-label"><BookOpen size={16}/> {filename}</div>
          <span>AI answers grounded in your uploaded document</span>
        </div>
        <div className="mode-buttons">
          <button onClick={summarize} disabled={loading || summaryLoading}>
            {summaryLoading ? <Loader2 className="spin" size={15}/> : <Sparkles size={15}/>} Summary
          </button>
          <button onClick={() => ask(factualPrompt)} disabled={loading || summaryLoading}><Search size={15}/> Factual</button>
        </div>
      </div>

      <div className="messages">
        {messages.length === 0 && (
          <div className="empty-chat">
            <Sparkles size={32}/>
            <h3>What would you like to know?</h3>
            <p>Ask about the methodology, results, conclusions, or any specific section.</p>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`message-row ${m.role}`}>
            <div className="message-bubble">
              {m.text.split("\n").map((line, j) => <p key={j}>{line || "\u00a0"}</p>)}
              {m.model && <div className="model-note">Generated locally with {m.model}</div>}
              {m.sources?.length > 0 && (
                <div className="sources">
                  <b>Retrieved passages</b>
                  {m.sources.map((s, j) => <div className="source" key={j}>#{j+1} · {(s.text || "").slice(0, 220)}…</div>)}
                </div>
              )}
            </div>
          </div>
        ))}
        {(loading || summaryLoading) && <div className="message-row assistant"><div className="message-bubble typing"><Loader2 className="spin" size={18}/> Generating with local AI…</div></div>}
      </div>

      <div className="composer">
        <input value={question} onChange={e => setQuestion(e.target.value)}
          onKeyDown={e => e.key === "Enter" && ask()}
          placeholder="Ask anything about your document..." disabled={loading || summaryLoading} />
        <button onClick={() => ask()} disabled={!question.trim() || loading || summaryLoading}><Send size={18}/></button>
      </div>
    </div>
  );
}
