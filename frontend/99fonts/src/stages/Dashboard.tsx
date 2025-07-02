import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { url_base } from '../utils';

interface FontRun {
  id: string;
  name: string;
  status: 'in_progress' | 'completed' | 'failed';
  created_at: string;
  updated_at: string;
  prompt?: string;
  download_url?: string;
}

function Dashboard() {
  const navigate = useNavigate();
  const [fontRuns, setFontRuns] = useState<FontRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [userName, setUserName] = useState('');

  useEffect(() => {
    const token = localStorage.getItem('authToken');
    if (!token) {
      navigate('/login');
      return;
    }
    
    fetchFontRuns();
    fetchUserProfile();
  }, []);

  const fetchUserProfile = async () => {
    try {
      const token = localStorage.getItem('authToken');
      const response = await fetch(`${url_base}/api/auth/profile`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        setUserName(data.name);
      }
    } catch (err) {
      console.error('Error fetching profile:', err);
    }
  };

  const fetchFontRuns = async () => {
    try {
      const token = localStorage.getItem('authToken');
      const response = await fetch(`${url_base}/api/font-runs`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        setFontRuns(data.runs || []);
      } else if (response.status === 401) {
        // Token expired or invalid
        localStorage.removeItem('authToken');
        localStorage.removeItem('userId');
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
    localStorage.removeItem('authToken');
    localStorage.removeItem('userId');
    navigate('/');
  };

  const startNewProject = async () => {
    try {
      const token = localStorage.getItem('authToken');
      const response = await fetch(`${url_base}/api/font-runs`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          name: `Font Project ${new Date().toLocaleDateString()}`
        })
      });

      if (response.ok) {
        const data = await response.json();
        window.url_extension = data.url_extension;
        navigate('/images');
      } else {
        setError('Failed to start new project');
      }
    } catch (err: any) {
      setError('Network error starting project');
      console.error('Error starting project:', err);
    }
  };

  const getStatusBadge = (status: string) => {
    const statusClasses = {
      'in_progress': 'status-badge status-progress',
      'completed': 'status-badge status-completed',
      'failed': 'status-badge status-failed'
    };
    
    const statusTexts = {
      'in_progress': 'In Progress',
      'completed': 'Completed',
      'failed': 'Failed'
    };

    return (
      <span className={statusClasses[status as keyof typeof statusClasses]}>
        {statusTexts[status as keyof typeof statusTexts]}
      </span>
    );
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

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
          <h1>Welcome back{userName ? `, ${userName}` : ''}!</h1>
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
          <div className="font-runs-grid">
            {fontRuns.map((run) => (
              <div key={run.id} className="font-run-card">
                <div className="card-header">
                  <h3>{run.name}</h3>
                  {getStatusBadge(run.status)}
                </div>
                
                <div className="card-content">
                  {run.prompt && (
                    <p className="font-prompt">"{run.prompt}"</p>
                  )}
                  <p className="font-date">
                    Created: {formatDate(run.created_at)}
                  </p>
                  {run.status === 'completed' && run.download_url && (
                    <a 
                      href={run.download_url} 
                      className="download-link"
                      download
                    >
                      Download Font
                    </a>
                  )}
                </div>
                
                <div className="card-actions">
                  {run.status === 'in_progress' && (
                    <button 
                      className="continue-button"
                      onClick={() => {
                        window.url_extension = run.id;
                        navigate('/images');
                      }}
                    >
                      Continue
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default Dashboard; 