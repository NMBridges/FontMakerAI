import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';

// Import stage components
import PromptStage from './stages/PromptStage';
import ImagesStage from './stages/ImagesStage';
import VectorizeStage from './stages/VectorizeStage';
import FontFileStage from './stages/FontFileStage';
import NavBar from './stages/NavBar';

// Initialize url_extension as a global variable
declare global {
  interface Window {
    url_extension: string;
  }
}

// Ensure the property exists on window
if (typeof window !== 'undefined' && typeof window.url_extension === 'undefined') {
  window.url_extension = '';
}

function App() {
  return (
    <Router>
      <div className="app-container">
        <NavBar />
        <div className="content-container">
          <Routes>
            <Route path="/" element={<PromptStage />} />
            <Route path="/images" element={<ImagesStage />} />
            <Route path="/vectorize" element={<VectorizeStage />} />
            <Route path="/font-file" element={<FontFileStage />} />
          </Routes>
        </div>
        <p className="read-the-docs">
          Contact glyphpy@gmail.com to learn more.
        </p>
      </div>
    </Router>
  );
}

export default App;
