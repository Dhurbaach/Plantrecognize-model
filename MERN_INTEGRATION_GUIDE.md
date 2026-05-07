# MERN Stack Integration Guide - Plant Identification Model

## Overview
This guide shows how to integrate your trained MobileNetV2 plant identification model into a MERN (MongoDB, Express, React, Node.js) application.

---

## Architecture

```
React Frontend (Upload Image)
         ↓
Express Backend (REST API)
         ↓
Python Script (TensorFlow Prediction)
         ↓
Trained Model + Class Names
         ↓
JSON Response (Plant Name + Confidence)
```

---

## Setup Instructions

### 1. Backend Setup (Express)

#### Step 1: Create backend folder
```bash
mkdir plant-api-backend
cd plant-api-backend
npm init -y
```

#### Step 2: Install dependencies
```bash
npm install express multer cors dotenv axios
npm install --save-dev nodemon
```

#### Step 3: Copy backend files
- Copy `backend_plant_api.js` to your backend folder
- Copy `predict.py` to the same directory
- Ensure `final_mobilenet_model.h5` and `class_indices.json` are in this directory

#### Step 4: Update package.json scripts
```json
{
  "scripts": {
    "start": "node backend_plant_api.js",
    "dev": "nodemon backend_plant_api.js"
  }
}
```

#### Step 5: Create .env file
```
PORT=5000
NODE_ENV=development
```

#### Step 6: Start backend
```bash
npm run dev
```

Your API will be available at `http://localhost:5000`

---

### 2. Frontend Setup (React)

#### Step 1: In your React project, copy the component
```bash
# If you have a React app already created
cp PlantIdentifier.jsx src/components/
cp PlantIdentifier.css src/components/
```

#### Step 2: Install axios (if not already installed)
```bash
npm install axios
```

#### Step 3: Create .env file in React root
```
REACT_APP_API_URL=http://localhost:5000
```

#### Step 4: Use component in your app
```jsx
// App.js or any page component
import PlantIdentifier from './components/PlantIdentifier';

function App() {
  return (
    <div className="App">
      <PlantIdentifier />
    </div>
  );
}

export default App;
```

#### Step 5: Start React dev server
```bash
npm start
```

---

### 3. Running Both Together

#### Terminal 1 (Backend):
```bash
cd plant-api-backend
npm run dev
```
Expected: `Plant identification API running on port 5000`

#### Terminal 2 (Frontend):
```bash
cd plant-detection-app
npm start
```
Expected: React app opens at `http://localhost:3000`

---

## API Endpoints

### 1. Health Check
```
GET /api/health
Response: { "status": "Model API is running" }
```

### 2. Plant Identification
```
POST /api/predict
Content-Type: multipart/form-data

Body:
  - image: <File> (JPEG, PNG, or GIF)

Response (Success):
{
  "success": true,
  "plant_name": "Tulsi",
  "confidence": 0.96,
  "confidence_percentage": "96.00"
}

Response (Error):
{
  "success": false,
  "error": "No image uploaded"
}
```

---

## Testing the API with cURL

```bash
# Test health endpoint
curl http://localhost:5000/api/health

# Test prediction (replace path/to/image.jpg with your image)
curl -X POST -F "image=@path/to/image.jpg" \
  http://localhost:5000/api/predict
```

---

## Frontend Usage

1. Click "Choose Plant Image" to select an image
2. See preview of selected image
3. Click "Identify Plant 🔍" to get prediction
4. View results with confidence percentage
5. Click "Clear ✕" to start over

---

## Environment Variables

### Backend (.env)
```
PORT=5000
NODE_ENV=development
```

### Frontend (.env)
```
REACT_APP_API_URL=http://localhost:5000
```

For production, update to your deployed backend URL:
```
REACT_APP_API_URL=https://your-api.com
```

---

## Troubleshooting

### Issue: "Error: Cannot find module 'tensorflow'"
**Solution**: Make sure Python is installed and TensorFlow is available:
```bash
pip install tensorflow numpy pillow
```

### Issue: "Model file not found"
**Solution**: Ensure `final_mobilenet_model.h5` and `class_indices.json` are in the backend folder

### Issue: CORS error
**Solution**: Already handled in `backend_plant_api.js`. For production, update:
```javascript
app.use(cors({
  origin: 'https://your-frontend-domain.com'
}));
```

### Issue: Python script not executing
**Solution**: Make predict.py executable:
```bash
chmod +x predict.py
```

---

## Production Deployment

### Option 1: Deploy with Heroku

**Backend:**
```bash
# Create Procfile
echo "web: node backend_plant_api.js" > Procfile

heroku create your-plant-api
git push heroku main
```

**Frontend:**
```bash
# Update .env for production
REACT_APP_API_URL=https://your-plant-api.herokuapp.com

npm run build
# Deploy using Netlify, Vercel, or Heroku
```

### Option 2: Docker Deployment

**Backend Dockerfile:**
```dockerfile
FROM node:18
WORKDIR /app
COPY . .
RUN npm install
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip install tensorflow numpy pillow
EXPOSE 5000
CMD ["npm", "start"]
```

---

## Performance Tips

1. **Image Validation**: Validate image size/format before upload
2. **Caching**: Cache predictions for identical images
3. **Rate Limiting**: Add rate limiting to prevent abuse
4. **Model Optimization**: Use TensorFlow Lite for faster inference
5. **Async Processing**: Use job queues for batch predictions

---

## Next Steps

- Add image history/database storage
- Implement user authentication
- Add confidence threshold alerts
- Create analytics dashboard
- Deploy to production
