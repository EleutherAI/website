import React from 'react';
import { createRoot } from 'react-dom/client';
import BasinExplorer from './BasinExplorer.jsx';

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BasinExplorer />
  </React.StrictMode>
);
