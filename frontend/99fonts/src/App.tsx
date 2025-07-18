import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import './App.css';

// Import stage components
import Landing from './stages/Landing';
import Login from './stages/Login';
import Signup from './stages/Signup';
import Dashboard from './stages/Dashboard';
import WorkflowPage from './stages/WorkflowPage';
import NavBar from './stages/NavBar';

// Initialize url_extension as a global variable
declare global {
  interface Window {
    url_extension: string;
    fontRunId: string;
  }
}

// Ensure the properties exist on window
if (typeof window !== 'undefined') {
  if (typeof window.url_extension === 'undefined') {
    window.url_extension = '';
  }
  if (typeof window.fontRunId === 'undefined') {
    window.fontRunId = '';
  }
}

function AppContent() {
  const location = useLocation();
  
  // Determine which navigation to show
  const isWorkflowPage = location.pathname === '/create-font';
  const isPublicPage = ['/', '/login', '/signup'].includes(location.pathname);
  
  const showNoNav = isPublicPage || location.pathname === '/dashboard' || isWorkflowPage;

  return (
    <div className="app-container">
      {!showNoNav && <NavBar />}
      
      <div className="content-container">
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/create-font" element={<WorkflowPage />} />
        </Routes>
      </div>
      
      {!isPublicPage && !isWorkflowPage && location.pathname !== '/dashboard' && (
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
