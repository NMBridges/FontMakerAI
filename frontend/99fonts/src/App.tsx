import { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Link, useNavigate, useLocation } from 'react-router-dom'
import fontsLogo from '/99fonts.svg'
import './App.css'

const url_base = 'http://44.210.86.218';
let url_extension = '';

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
      const data = await (await fetch(`${url_base}/api/sample_diffusion`, {headers: {'Content-Type': 'application/json'}})).json();
      url_extension = data.url_extension;
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
  const [progress, setProgress] = useState(0);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  
  useEffect(() => {
    const checkProgress = async () => {
      try {
        if (!url_extension) {
          console.log('No URL extension found');
          return;
        }
        const response = await fetch(`${url_base}${url_extension}`);
        if (!response.ok) {
          console.log('Response not ok');
          return;
        }
        let data;
        try {
          data = await response.json();
          console.log(data);
        } catch (err) {
          // If JSON parsing fails, we likely received an image
          console.log('Received image response');
          clearInterval(intervalId);
          setImageUrl(`${url_base}${url_extension}`);
          return;
        }
        
        if (data && data.progress) {
          setProgress(data.progress);
        }
      } catch (err) {
        console.error("Error checking progress:", err);
      }
    };
    
    // Start polling every 500ms
    const intervalId = setInterval(checkProgress, 500);
    
    // Clean up interval on component unmount
    return () => clearInterval(intervalId);
  }, []);
  
  return (
    <div className="stage-content">
      <h2>Select Font Images</h2>
      <div className="images-container">
        {imageUrl ? (
          <img src={imageUrl} alt="Generated font" />
        ) : (
          <div>
            <p>Generating font images... {typeof progress === 'number' ? `${Math.round(progress * 100 / 32)}%` : progress}</p>
            <div className="progress-bar">
              <div 
                className="progress-fill" 
                style={{ width: `${typeof progress === 'number' ? progress * 100 / 32 : 0}%` }}
              ></div>
            </div>
          </div>
        )}
      </div>
      <button 
        onClick={() => navigate('/vectorize')}
        disabled={!imageUrl}
      >
        Convert to Vector Paths
      </button>
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
