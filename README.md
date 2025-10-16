# Mony

Visualize multiple UI directions from a single brief. Describe the product experience, pick a designer style, and generate prompts for OpenRouter-compatible image models.

## Getting started

### Prerequisites
- [Node.js](https://nodejs.org/) 18+
- An [OpenRouter](https://openrouter.ai) API key (add it to a local `.env` file)

### Installation
```bash
npm install --prefix client
npm install --prefix server
```

### Environment variables
Copy the provided examples and add your secrets:

```bash
cp .env.example .env
cp client/.env.example client/.env # optional, only needed when overriding the API URL
```

Update `.env` with your `OPENROUTER_API_KEY` and optional metadata.

### Running the app locally
In two separate terminals run:

```bash
npm run dev --prefix server
npm run dev --prefix client
```

The Vite dev server proxies `/api` requests to the Express API. The UI is available at http://localhost:5173.

## How it works
- Prompts for the Conservative, Modern, and Funky designers automatically incorporate the product description you provide.
- You can fine-tune each prompt directly in the UI before sending the request.
- Clicking **Generate visual concepts** calls the backend, which forwards prompts to OpenRouter’s image generation endpoint and returns the image URLs.

Feel free to tweak the preset prompts or add new designer personas to explore additional art directions.
