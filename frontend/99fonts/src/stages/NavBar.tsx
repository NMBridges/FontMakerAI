import { Link, useLocation } from 'react-router-dom';

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

export default NavBar; 