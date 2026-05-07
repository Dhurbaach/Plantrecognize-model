# Text-to-Speech (TTS) Setup Guide

This guide explains how to set up the text-to-speech feature for your Plant Identification app, including support for Nepali language.

## Features

- **English TTS**: Works automatically using the browser's built-in speech synthesis
- **Nepali TTS**: Uses backend API for high-quality Nepali speech

## Two-Tier Approach

### 1. **Free Tier (No Setup Required)**
- Uses **Google Translate TTS API** as fallback
- Works for both English and Nepali
- No authentication needed
- May have rate limiting

### 2. **Premium Tier (Recommended)**
- Uses **Google Cloud Text-to-Speech API**
- Higher quality voices
- Supports multiple accents and speaking styles
- Requires setup (see below)

## Setup Google Cloud Text-to-Speech (Optional)

### Step 1: Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click **"Create Project"** or select an existing project
3. Enter a project name (e.g., "Plant ID TTS")
4. Click **Create**

### Step 2: Enable the Text-to-Speech API

1. In the Cloud Console, go to **APIs & Services → Library**
2. Search for **"Text-to-Speech"**
3. Click on it and click **Enable**

### Step 3: Create a Service Account

1. Go to **APIs & Services → Credentials**
2. Click **Create Credentials → Service Account**
3. Fill in the service account details:
   - Service account name: `plant-id-tts`
   - Click **Create and Continue**
4. Grant role: Select **Editor** (or **Cloud Text-to-Speech API Admin**)
5. Click **Continue** and **Done**

### Step 4: Create and Download Service Account Key

1. Click on the service account you just created
2. Go to **Keys** tab
3. Click **Add Key → Create new key**
4. Choose **JSON** format
5. Click **Create** - this downloads your credentials JSON file

### Step 5: Configure Backend

1. Copy the downloaded JSON file to your backend directory:
   ```
   cp ~/Downloads/your-service-account-key.json ./Herboscope-main/backend/credentials.json
   ```

2. Update your backend `.env` file:
   ```env
   GOOGLE_APPLICATION_CREDENTIALS=./credentials.json
   ```

### Step 6: Install Dependencies

In the backend directory:

```bash
cd Herboscope-main/backend
npm install
```

This installs the Google Cloud Text-to-Speech SDK.

### Step 7: Restart Backend

```bash
npm run dev
```

The server will log: `"Google Cloud Text-to-Speech is configured"` when ready.

## Usage

1. Open the plant identification results page
2. Click the **NE** button to switch to Nepali
3. Click **🔊 Read** 
4. The app will:
   - Translate the description to Nepali
   - Request audio from the backend TTS service
   - Play the Nepali audio

## Troubleshooting

### "Nepali voice not available" Error
- **Solution 1**: Set up Google Cloud TTS (see above)
- **Solution 2**: Install Nepali language pack on your OS
  - **Windows**: Settings → Time & Language → Language & region → Add language → नेपाली
  - **Mac**: System Preferences → Accessibility → Speech → Select Nepali voice

### TTS is slow
- Google Cloud has warm-up time on first use
- Subsequent requests are faster (responses are cached)

### "No audio data in response"
- Backend TTS service may be down
- Check browser console for error details
- Restart backend: `npm run dev`

### Fallback API not working
- Google Translate API has rate limits
- Wait a few minutes and try again
- Set up Google Cloud TTS for reliable service

## Cost Considerations

- **Google Translate TTS**: Free (no quota)
- **Google Cloud TTS**: $16 per 1 million characters after free quota
  - Free tier: 1 million characters/month

## Environment Variables

Create/update `.env` in the backend directory:

```env
PORT=3000
CLIENT_URL=http://localhost:5173
GOOGLE_APPLICATION_CREDENTIALS=./credentials.json
MONGODB_URI=your_mongodb_uri
```

## Testing

To test the TTS endpoint manually:

```bash
curl -X POST http://localhost:3000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "नेपाली भाषा परीक्षा",
    "language": "ne-NP"
  }'
```

## Support

For issues with Google Cloud setup, visit:
- [Google Cloud Text-to-Speech Documentation](https://cloud.google.com/text-to-speech/docs)
- [Service Account Setup Guide](https://cloud.google.com/docs/authentication/getting-started)
