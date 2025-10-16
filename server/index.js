const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');

dotenv.config({ path: '../.env' });

const app = express();
const PORT = process.env.PORT || 4000;

app.use(cors());
app.use(express.json());

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const OPENROUTER_APP_TITLE = process.env.OPENROUTER_APP_TITLE || 'Mony UI Visualizer';
const OPENROUTER_REFERRER = process.env.OPENROUTER_REFERRER || 'http://localhost:5173';

if (!OPENROUTER_API_KEY) {
  console.warn('Warning: OPENROUTER_API_KEY is not set. Image generation requests will fail.');
}

async function requestImage(prompt, size = '1024x1024') {
  if (!OPENROUTER_API_KEY) {
    throw new Error('Missing OpenRouter API key.');
  }

  const response = await fetch('https://openrouter.ai/api/v1/images', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${OPENROUTER_API_KEY}`,
      'HTTP-Referer': OPENROUTER_REFERRER,
      'X-Title': OPENROUTER_APP_TITLE,
    },
    body: JSON.stringify({
      model: 'deepai/flux-1.1-pro',
      prompt,
      size,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`OpenRouter error: ${response.status} ${errorText}`);
  }

  const data = await response.json();
  if (!data?.data?.[0]?.url) {
    throw new Error('Unexpected OpenRouter response.');
  }

  return data.data[0].url;
}

app.post('/api/generate', async (req, res) => {
  const { prompts, size } = req.body || {};

  if (!Array.isArray(prompts) || prompts.length === 0) {
    return res.status(400).json({ error: 'No prompts provided.' });
  }

  try {
    const results = [];

    for (const promptConfig of prompts) {
      const { id, label, prompt } = promptConfig;
      if (!prompt || typeof prompt !== 'string') {
        throw new Error(`Invalid prompt for designer ${label || id || 'unknown'}`);
      }

      const imageUrl = await requestImage(prompt, size);
      results.push({ id, label, prompt, imageUrl });
    }

    res.json({ results });
  } catch (error) {
    console.error('Image generation failed:', error);
    res.status(500).json({ error: error.message || 'Failed to generate images.' });
  }
});

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok' });
});

app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});
