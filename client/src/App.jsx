import { useEffect, useMemo, useState } from 'react';
import './App.css';

const DESIGNER_PRESETS = [
  {
    id: 'conservative',
    label: 'Conservative',
    blurb: 'Timeless design language with balanced typography and subtle color palettes.',
    template:
      'Design a conservative, enterprise-ready user interface for {description}. Emphasize clarity, ample whitespace, and structured information hierarchy. Showcase a complete screen with refined typography, muted colors, and polished UI kit elements.',
  },
  {
    id: 'modern',
    label: 'Modern',
    blurb: 'Sleek and product-focused with bold contrast, layered cards, and crisp visuals.',
    template:
      'Create a modern, product-focused UI for {description}. Use bold gradients, layered cards, and confident typography. Include hero sections, interactive states, and micro-interactions that feel production ready.',
  },
  {
    id: 'funky',
    label: 'Funky',
    blurb: 'Playful art direction, vibrant colors, unexpected shapes, and expressive type choices.',
    template:
      'Illustrate a funky, experimental interface for {description}. Lean into expressive typography, vibrant color blocking, organic shapes, and unexpected layout choices that feel like a creative concept board.',
  },
];

const DEFAULT_DESCRIPTION = 'a personal finance dashboard that highlights spending, savings goals, and upcoming invoices for freelancers';

const applyTemplate = (template, description) => {
  const safeDescription = description && description.trim().length > 0 ? description.trim() : 'the requested product experience';
  return template.replaceAll('{description}', safeDescription);
};

const buildInitialState = (description = DEFAULT_DESCRIPTION) =>
  DESIGNER_PRESETS.map((preset) => ({
    ...preset,
    prompt: applyTemplate(preset.template, description),
    isCustom: false,
  }));

const API_BASE_URL = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '');

function App() {
  const [description, setDescription] = useState(DEFAULT_DESCRIPTION);
  const [size, setSize] = useState('1024x1024');
  const [designers, setDesigners] = useState(() => buildInitialState(DEFAULT_DESCRIPTION));
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const hasDescription = useMemo(() => description.trim().length > 0, [description]);

  useEffect(() => {
    setDesigners((prev) =>
      prev.map((designer) =>
        designer.isCustom
          ? designer
          : {
              ...designer,
              prompt: applyTemplate(designer.template, description),
            }
      )
    );
  }, [description]);

  const handlePromptChange = (id, nextPrompt) => {
    setDesigners((prev) =>
      prev.map((designer) =>
        designer.id === id
          ? {
              ...designer,
              prompt: nextPrompt,
              isCustom: nextPrompt.trim() !== applyTemplate(designer.template, description).trim(),
            }
          : designer
      )
    );
  };

  const handleResetPrompt = (id) => {
    setDesigners((prev) =>
      prev.map((designer) =>
        designer.id === id
          ? {
              ...designer,
              prompt: applyTemplate(designer.template, description),
              isCustom: false,
            }
          : designer
      )
    );
  };

  const handleGenerate = async () => {
    setError('');
    setIsLoading(true);
    setResults([]);

    try {
      const endpoint = API_BASE_URL ? `${API_BASE_URL}/api/generate` : '/api/generate';
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          size,
          prompts: designers.map(({ id, label, prompt }) => ({ id, label, prompt })),
        }),
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || 'Request failed');
      }

      const data = await response.json();
      setResults(Array.isArray(data.results) ? data.results : []);
    } catch (err) {
      setError(err.message || 'Failed to generate concepts.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="hero">
        <h1>Mony – UI Moodboard Generator</h1>
        <p>
          Describe the interface you want to build, pick the creative direction, and instantly turn
          those ideas into visual prompts for your favorite image model via OpenRouter.
        </p>
      </header>

      <section className="panel">
        <label className="field">
          <span>What are we designing?</span>
          <textarea
            placeholder='Example: "A personal finance dashboard that highlights spending insights and savings goals for young professionals."'
            value={description}
            onChange={(event) => setDescription(event.target.value)}
            rows={4}
          />
        </label>

        <div className="field inline">
          <label>
            <span>Image size</span>
            <select value={size} onChange={(event) => setSize(event.target.value)}>
              <option value="1024x1024">1024 × 1024</option>
              <option value="768x1024">768 × 1024</option>
              <option value="1024x768">1024 × 768</option>
            </select>
          </label>
          <div className="hint">
            Prompts update automatically when you edit the description. Tweak any prompt to give a
            designer more personality.
          </div>
        </div>
      </section>

      <section className="designer-grid">
        {designers.map((designer) => (
          <article key={designer.id} className="designer-card">
            <header>
              <h2>{designer.label}</h2>
              <p>{designer.blurb}</p>
            </header>
            <label className="field">
              <span>Prompt sent to OpenRouter</span>
              <textarea
                value={designer.prompt}
                onChange={(event) => handlePromptChange(designer.id, event.target.value)}
                rows={7}
              />
            </label>
            <div className="card-actions">
              <button type="button" onClick={() => handleResetPrompt(designer.id)}>
                Reset to template
              </button>
              {designer.isCustom && <span className="badge">customized</span>}
            </div>
          </article>
        ))}
      </section>

      <section className="actions">
        <button type="button" className="primary" onClick={handleGenerate} disabled={isLoading || !hasDescription}>
          {isLoading ? 'Generating concepts…' : 'Generate visual concepts'}
        </button>
        {!hasDescription && <p className="error">Add a short description to unlock generation.</p>}
        {error && <p className="error">{error}</p>}
      </section>

      <section className="results">
        {results.length > 0 && <h2>Concept gallery</h2>}
        <div className="results-grid">
          {results.map((result) => (
            <figure key={result.id} className="result-card">
              <div className="image-frame">
                {result.imageUrl ? (
                  <img src={result.imageUrl} alt={`${result.label} concept`} loading="lazy" />
                ) : (
                  <div className="image-placeholder">No preview returned</div>
                )}
              </div>
              <figcaption>
                <h3>{result.label}</h3>
                <details>
                  <summary>Prompt</summary>
                  <p>{result.prompt}</p>
                </details>
                {result.imageUrl && (
                  <a href={result.imageUrl} target="_blank" rel="noreferrer">
                    Open full image ↗
                  </a>
                )}
              </figcaption>
            </figure>
          ))}
        </div>
      </section>

      <footer className="footer">
        <p>
          Configure your <code>.env</code> with <code>OPENROUTER_API_KEY</code> before generating. The
          prompts can be copy-pasted into any image workflow if you prefer to experiment manually.
        </p>
      </footer>
    </div>
  );
}

export default App;
