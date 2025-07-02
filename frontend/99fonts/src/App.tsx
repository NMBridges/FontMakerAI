import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import './App.css';

// Import stage components
import Landing from './stages/Landing';
import Login from './stages/Login';
import Signup from './stages/Signup';
import Dashboard from './stages/Dashboard';
import PromptStage from './stages/PromptStage';
import ImagesStage from './stages/ImagesStage';
import VectorizeStage from './stages/VectorizeStage';
import FontFileStage from './stages/FontFileStage';
import NavBar from './stages/NavBar';
import AuthenticatedNavBar from './stages/AuthenticatedNavBar';

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

function AppContent() {
  const location = useLocation();
  
  // Determine which navigation to show
  const isAuthWorkflow = ['/images', '/vectorize', '/font-file'].includes(location.pathname);
  const isPublicPage = ['/', '/login', '/signup'].includes(location.pathname);
  
  const showAuthenticatedNav = isAuthWorkflow;
  const showNoNav = isPublicPage || location.pathname === '/dashboard';

  return (
    <div className="app-container">
      {showAuthenticatedNav && <AuthenticatedNavBar />}
      {!showNoNav && !showAuthenticatedNav && <NavBar />}
      
      <div className="content-container">
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/images" element={<ImagesStage />} />
          <Route path="/vectorize" element={<VectorizeStage />} />
          <Route path="/font-file" element={<FontFileStage />} />
        </Routes>
      </div>
      
      {!isPublicPage && !isAuthWorkflow && location.pathname !== '/dashboard' && (
        <p className="read-the-docs">
          Contact glyphpy@gmail.com to learn more.
        </p>
      )}
    </div>
  );
}

function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  );
}

export default App;
