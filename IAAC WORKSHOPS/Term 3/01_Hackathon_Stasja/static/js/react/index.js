import React from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter, Routes, Route } from 'react-router-dom';

// Import our components
import App from './components/App';
import ModelViewer from './components/ModelViewer';

// Mount React App
const container = document.getElementById('react-app');
const root = createRoot(container);

root.render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/view/:filename" element={<ModelViewer />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);