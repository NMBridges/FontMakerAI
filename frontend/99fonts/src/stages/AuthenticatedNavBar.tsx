import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';

function AuthenticatedNavBar() {
  const location = useLocation();
  const navigate = useNavigate();
  const { logout } = useAuth();
  
  // Determine which step is active based on the current path
  const isCompleted = (path: string): boolean => {
    const routes = ['/prompt', '/images', '/vectorize', '/font-file'];
    const currentIndex = routes.indexOf(location.pathname);
    const pathIndex = routes.indexOf(path);
    
    return pathIndex <= currentIndex;
  };

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  const handleBackToDashboard = () => {
    navigate('/dashboard');
  };

  return (
    <div className="navbar">
      <div className="navbar-header">
        <button onClick={handleBackToDashboard} className="back-to-dashboard">
          ← Dashboard
        </button>
        <button onClick={handleLogout} className="logout-button">
          Logout
        </button>
      </div>
      
      <div className="progress-steps-container">
        <div className="progress-step-item">
          <Link to="/prompt" className="step-link">
            <div className={`progress-line ${isCompleted('/prompt') ? 'completed' : ''}`}></div>
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

export default AuthenticatedNavBar; 