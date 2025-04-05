import { useState } from 'react'
import { BrowserRouter as Router, Routes, Route, Link, useNavigate, useLocation } from 'react-router-dom'
import fontsLogo from '/99fonts.svg'
import './App.css'

// Prompt Stage Component
function PromptStage() {
  const navigate = useNavigate();

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      handleClick();
    }
  };
  
  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const target = e.target as HTMLTextAreaElement;
    target.style.height = "auto";
    target.style.height = `${target.scrollHeight}px`;
  }

  const handleClick = async () => {
    try {
      navigate('/images');
      const data = await (await fetch('http://localhost:8080/sample_diffusion', {headers: {'Content-Type': 'application/json'}})).json();
      console.log(data);
    } catch (err: any) {
      console.log(err.message);
    }
  }

  return (
    <div className="stage-content">
      <h1 style={{marginTop: '3em'}}>Copyright-free, custom fonts. Powered by AI.</h1>
      <div className="search-bar-container">
        <textarea className="search-bar" placeholder="Describe your next font" onInput={handleInput} onKeyDown={handleKeyDown}></textarea>
        <button className="generate-button" onClick={handleClick}>Generate</button> 
      </div>
    </div>
  );
}

// Images Stage Component
function ImagesStage() {
  const navigate = useNavigate();
  
  return (
    <div className="stage-content">
      <h2>Select Font Images</h2>
      <div className="images-container">
        {/* Images will be displayed here */}
        <p>Font images will appear here</p>
      </div>
      <button onClick={() => navigate('/vectorize')}>Continue to Vectorize</button>
    </div>
  );
}

// Vectorize Stage Component
function VectorizeStage() {
  const navigate = useNavigate();
  
  return (
    <div className="stage-content">
      <h2>Vectorize Your Font</h2>
      <div className="vectorize-container">
        <p>Vectorization tools will appear here</p>
      </div>
      <button onClick={() => navigate('/font-file')}>Generate Font File</button>
    </div>
  );
}

// Font File Stage Component
function FontFileStage() {
  return (
    <div className="stage-content">
      <h2>Your Font File</h2>
      <div className="font-file-container">
        <p>Download your font file here</p>
        <button>Download Font</button>
      </div>
    </div>
  );
}

// Navigation Bar Component
function NavBar() {
  const location = useLocation();
  
  // Determine which step is active based on the current path
  const isCompleted = (path: string): boolean => {
    const routes = ['/', '/images', '/vectorize', '/font-file'];
    const currentIndex = routes.indexOf(location.pathname);
    const pathIndex = routes.indexOf(path);
    
    return pathIndex <= currentIndex;
  };

  return (
    <div className="navbar">
      <div className="progress-steps-container">
        <div className="progress-step-item">
          <Link to="/" className="step-link">
            <div className={`progress-line ${isCompleted('/') ? 'completed' : ''}`}></div>
            <div className="step-text">Describe font</div>
          </Link>
        </div>
        <div className="progress-step-item">
          <Link to="/images" className="step-link">
            <div className={`progress-line ${isCompleted('/images') ? 'completed' : ''}`}></div>
            <div className="step-text">View bitmap glyphs</div>
          </Link>
        </div>
        <div className="progress-step-item">
          <Link to="/vectorize" className="step-link">
            <div className={`progress-line ${isCompleted('/vectorize') ? 'completed' : ''}`}></div>
            <div className="step-text">View vectorized glyphs</div>
          </Link>
        </div>
        <div className="progress-step-item">
          <Link to="/font-file" className="step-link">
            <div className={`progress-line ${isCompleted('/font-file') ? 'completed' : ''}`}></div>
            <div className="step-text">Download font</div>
          </Link>
        </div>
      </div>
    </div>
  );
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
  )
}

export default App
