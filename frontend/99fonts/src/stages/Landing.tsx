import { Link, useNavigate } from 'react-router-dom';
import { useEffect } from 'react';
import { useAuth } from '../hooks/useAuth';

function Landing() {
  const navigate = useNavigate();
  const { isAuthenticated, isLoading } = useAuth();

  useEffect(() => {
    if (isAuthenticated) {
      navigate('/dashboard');
    }
  }, [isAuthenticated, navigate]);

  if (isLoading) {
    return (
      <div className="landing-container">
        <div className="landing-content">
          <div className="loading">Checking authentication...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="landing-container">
      <div className="landing-content">
        <h1 className="landing-title">Copyright-free, custom fonts. Powered by AI.</h1>
        <p className="landing-subtitle">
          Create unique, personalized fonts using artificial intelligence. 
          Generate professional-quality typography for your projects in minutes.
        </p>
        
        <div className="landing-buttons">
          <Link to="/login" className="sign-in-button">
            Sign In
          </Link>
          <Link to="/signup" className="signup-button">
            Sign Up
          </Link>
        </div>
        
        <div className="landing-features">
          <div className="feature">
            <h3>AI-Powered</h3>
            <p>Advanced machine learning creates unique fonts from your descriptions</p>
          </div>
          <div className="feature">
            <h3>Copyright-Free</h3>
            <p>All generated fonts are completely original and royalty-free</p>
          </div>
          <div className="feature">
            <h3>Professional Quality</h3>
            <p>Production-ready fonts for web, print, and digital design</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Landing; 