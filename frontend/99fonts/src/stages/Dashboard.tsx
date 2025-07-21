import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { url_base } from '../utils';
import { useAuth } from '../hooks/useAuth';

interface FontRun {
  id: string;
  status: string;
  created_at: string;
  updated_at: string;
  prompt?: string;
  download_url?: string;
}

// Extend the global Window interface
declare global {
  interface Window {
    url_extension: string;
    fontRunId: string;
  }
}

function Dashboard() {
  const navigate = useNavigate();
  const { isAuthenticated, isLoading, user, logout } = useAuth();
  const [fontRuns, setFontRuns] = useState<FontRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    console.debug("Dashboard useEffect - isAuthenticated:", isAuthenticated, "isLoading:", isLoading);
    
    // Only redirect to login if we're not loading and not authenticated
    if (!isLoading && !isAuthenticated) {
      console.debug("Redirecting to login because not authenticated");
      navigate('/login');
      return;
    }
    
    // Only fetch font runs if we're authenticated and not loading
    if (isAuthenticated && !isLoading) {
      console.debug("Fetching font runs");
      fetchFontRuns();
    }
  }, [isAuthenticated, isLoading, navigate]);

  const fetchFontRuns = async () => {
    try {
      const token = localStorage.getItem('authToken');
      const response = await fetch(`${url_base}/api/dashboard/font-runs`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        setFontRuns(data.fontRuns || []);
      } else if (response.status === 401) {
        // Token expired or invalid
        logout();
        navigate('/login');
      } else {
        setError('Failed to load font runs');
      }
    } catch (err: any) {
      setError('Network error loading font runs');
      console.error('Error fetching font runs:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  // Start new project: make API call to create a new font run before navigating
  const startNewProject = async () => {
    try {
      const token = localStorage.getItem('authToken');
      const response = await fetch(`${url_base}/api/dashboard/create-font-run`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({})
      });

      if (!response.ok) {
        throw new Error('Failed to create font run');
      }

      const fontRunData = await response.json();
      console.debug('Created font run:', fontRunData);

      // Set global fontRunId from API response
      window.fontRunId = fontRunData.fontRunId;
      window.url_extension = ''; // Clear any previous url_extension
      
      navigate('/create-font');

    } catch (err) {
      console.error('Error creating new font run:', err);
      setError('Failed to create new font project. Please try again.');
      // Navigate anyway but clear the global variables
      window.fontRunId = '';
      window.url_extension = '';
      navigate('/create-font');
    }
  };

  const getStatusBadge = (status: string) => {
    // Map status to stage progress (X/4 format)
    const getStageProgress = (status: string) => {
      switch (status) {
        case 'EMPTY_DESCRIPTION':
          return { progress: '1/4', stage: 'Description', color: '#fbe8c6' };
        case 'IMAGES_GENERATING':
          return { progress: '2/4', stage: 'Images Generating', color: '#fff3cd' };
        case 'IMAGES_GENERATED':
          return { progress: '2/4', stage: 'Images Generated', color: '#f4f9c7' };
        case 'VECTORIZATION_STAGE':
          return { progress: '3/4', stage: 'Vectorization', color: '#c6f7e2' };
        case 'DOWNLOAD_STAGE':
          return { progress: '4/4', stage: 'Ready', color: '#b2ebf2' };
        default:
          return { progress: '0/4', stage: 'Unknown', color: '#bb0000' };
      }
    };

    const statusClasses = {
      'EMPTY_DESCRIPTION': 'status-badge status-empty',
      'IMAGES_GENERATING': 'status-badge status-generating',
      'IMAGES_GENERATED': 'status-badge status-completed',
      'VECTORIZATION_STAGE': 'status-badge status-progress',
      'DOWNLOAD_STAGE': 'status-badge status-download',
      'UNKNOWN': 'status-badge status-unknown'
    };

    const { progress, stage, color } = getStageProgress(status);
    
    console.log("status: ", status);
    const statusClass = statusClasses[status as keyof typeof statusClasses] || 'status-badge status-unknown';
    return (
      <span className={statusClass} style={{ backgroundColor: color }}>
        {progress} - {stage}
      </span>
    );
  };

  const formatRelativeTime = (dateString: string) => {
    const now_local = new Date();
    const date = new Date(dateString);
    const diffInSeconds = Math.floor((now_local.getTime() - date.getTime() + now_local.getTimezoneOffset() * 60000) / 1000);

    if (diffInSeconds < 60) {
      return `${diffInSeconds} seconds ago`;
    }

    const diffInMinutes = Math.floor(diffInSeconds / 60);
    if (diffInMinutes < 60) {
      return `${diffInMinutes} minute${diffInMinutes === 1 ? '' : 's'} ago`;
    }

    const diffInHours = Math.floor(diffInMinutes / 60);
    if (diffInHours < 24) {
      return `${diffInHours} hour${diffInHours === 1 ? '' : 's'} ago`;
    }

    const diffInDays = Math.floor(diffInHours / 24);
    if (diffInDays < 30) {
      return `${diffInDays} day${diffInDays === 1 ? '' : 's'} ago`;
    }

    const diffInMonths = Math.floor(diffInDays / 30);
    if (diffInMonths < 12) {
      return `${diffInMonths} month${diffInMonths === 1 ? '' : 's'} ago`;
    }

    const diffInYears = Math.floor(diffInMonths / 12);
    return `${diffInYears} year${diffInYears === 1 ? '' : 's'} ago`;
  };

  const getStageFromStatus = (status: string): number => {
    switch (status) {
      case 'EMPTY_DESCRIPTION':
        return 0;
      case 'IMAGES_GENERATING':
      case 'IMAGES_GENERATED':
        return 1;
      case 'VECTORIZATION_STAGE':
        return 2;
      case 'DOWNLOAD_STAGE':
        return 3;
      default:
        return 0;
    }
  };

  const handleFontRunClick = (run: FontRun) => {
    // Set global font run ID
    window.fontRunId = run.id;
    
    // Navigate to workflow page with stage information
    navigate('/create-font', { 
      state: { 
        fontRunId: run.id,
        startStage: getStageFromStatus(run.status),
        fontRunData: run
      }
    });
  };

  // Show loading while auth is being checked
  if (isLoading) {
    return (
      <div className="dashboard-container">
        <div className="loading">Checking authentication...</div>
      </div>
    );
  }

  // Show loading while font runs are being fetched
  if (loading) {
    return (
      <div className="dashboard-container">
        <div className="loading">Loading your projects...</div>
      </div>
    );
  }

  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <div className="header-content">
          <h1>Welcome back{user?.name ? `, ${user.name}` : ''}!</h1>
          <div className="header-actions">
            <button className="primary-button" onClick={startNewProject}>
              + Create New Font
            </button>
            <button className="secondary-button" onClick={handleLogout}>
              Logout
            </button>
          </div>
        </div>
      </div>

      <div className="dashboard-content">
        <h2>Your Font Projects</h2>
        
        {error && <div className="error-message">{error}</div>}
        
        {fontRuns.length === 0 ? (
          <div className="empty-state">
            <h3>No font projects yet</h3>
            <p>Create your first custom font by clicking the button above!</p>
          </div>
        ) : (
          <div className="font-runs-table-container">
            <table className="font-runs-table">
              <thead>
                <tr>
                  <th>Prompt</th>
                  <th>Status</th>
                  <th>Last Modified</th>
                </tr>
              </thead>
              <tbody>
                {fontRuns
                  .sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())
                  .map((run) => (
                    <tr key={run.id} className="font-run-row" onClick={() => handleFontRunClick(run)}>
                      <td className="font-prompt">
                        {run.prompt ? `"${run.prompt}"` : '-'}
                      </td>
                      <td className="font-status">
                        {getStatusBadge(run.status)}
                      </td>
                      <td className="font-date">
                        {formatRelativeTime(run.updated_at)}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

export default Dashboard; 