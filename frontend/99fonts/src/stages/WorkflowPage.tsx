import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { url_base } from '../utils';

// Import individual stage components (we'll convert them to sub-components)
import PromptSection from './workflow/PromptSection.tsx';
import ImagesSection from './workflow/ImagesSection.tsx';
import VectorizeSection from './workflow/VectorizeSection.tsx';
import FontFileSection from './workflow/FontFileSection.tsx';

// Declare the global variables
declare global {
  interface Window {
    url_extension: string;
    fontRunId: string;
  }
}

interface WorkflowState {
  currentStage: number;
  prompt: string;
  images: string[] | null;
  vectorizedImages: (string | null)[];
  vectorizationComplete: boolean[] | null;
  fontFileUrl: string | null;
}

type StageData = 
  | { prompt: string }
  | { images: string[] }
  | { vectorizedImages: (string | null)[] }
  | { fontFileUrl: string };

function WorkflowPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const { isAuthenticated, isLoading } = useAuth();
  
  // Get the stage information from navigation state
  const locationState = location.state as { 
    fontRunId?: string; 
    startStage?: number; 
    fontRunData?: any 
  } | null;
  
  const [workflowState, setWorkflowState] = useState<WorkflowState>({
    currentStage: locationState?.startStage ?? 0, // Use passed stage or default to 0
    prompt: locationState?.fontRunData?.prompt ?? '',
    images: null,
    vectorizedImages: Array(26).fill(null),
    vectorizationComplete: null,
    fontFileUrl: null
  });

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      navigate('/login');
    }
  }, [isAuthenticated, isLoading, navigate]);

  useEffect(() => {
    // If we have font run data from navigation, load any existing data
    if (locationState?.fontRunData && locationState?.fontRunId) {
      const fontRunData = locationState.fontRunData;
      setWorkflowState(prev => ({
        ...prev,
        currentStage: locationState.startStage ?? 0,
        prompt: fontRunData.prompt ?? prev.prompt
      }));
      
      // Fetch complete font run data from backend
      fetchFontRunData(locationState.fontRunId);
    }
  }, [locationState]);

  const fetchFontRunData = async (fontRunId: string) => {
    try {
      const token = localStorage.getItem('authToken');
      const response = await fetch(`${url_base}/api/fontrun/${fontRunId}/data`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        const result = await response.json();
        if (result.success && result.data) {
          const data = result.data;
          
          setWorkflowState(prev => ({
            ...prev,
            prompt: data.prompt || prev.prompt,
            images: data.images || prev.images,
            vectorizedImages: data.vectorizedImages || prev.vectorizedImages,
            vectorizationComplete: data.vectorizationComplete || prev.vectorizationComplete,
            fontFileUrl: data.fontFileUrl || prev.fontFileUrl,
            currentStage: Math.max(prev.currentStage, data.stage ?? 0)
          }));
        }
      } else {
        console.error('Failed to fetch font run data');
      }
    } catch (error) {
      console.error('Error fetching font run data:', error);
    }
  };

  const updateStageInBackend = async (stage: number) => {
    try {
      const token = localStorage.getItem('authToken');
      const response = await fetch(`${url_base}/api/fontrun/${window.fontRunId}/updateStage`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ 'stage': stage })
      });
      
      if (!response.ok) {
        console.error('Failed to update stage in backend');
      }
    } catch (error) {
      console.error('Error updating stage:', error);
    }
  };

  const handleStageComplete = (stage: number, data: StageData) => {
    setWorkflowState(prev => ({
      ...prev,
      currentStage: Math.max(prev.currentStage, stage + 1),
      ...data
    }));

    // Update backend stage when moving to vectorization (stage 4) or download (stage 5)
    if (stage === 2) { // Moving from Images to Vectorization
      updateStageInBackend(3); // Vectorization Stage
    } else if (stage === 3) { // Moving from Vectorization to Download
      updateStageInBackend(4); // Download Stage
    }
    
    // Scroll to next stage
    const nextStageElement = document.getElementById(`stage-${stage + 1}`);
    if (nextStageElement) {
      nextStageElement.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  };

  const handleBackToDashboard = () => {
    navigate('/dashboard');
  };

  const handleLogout = () => {
    localStorage.removeItem('authToken');
    localStorage.removeItem('userId');
    navigate('/');
  };

  if (isLoading) {
    return (
      <div className="workflow-container">
        <div className="loading">Checking authentication...</div>
      </div>
    );
  }

  return (
    <div className="workflow-container">
      {/* Workflow Header */}
      <div className="workflow-header">
        <div className="header-content">
          <button onClick={handleBackToDashboard} className="back-to-dashboard">
            ← Dashboard
          </button>
          <h1>Create Your Font</h1>
          <button onClick={handleLogout} className="logout-button">
            Logout
          </button>
        </div>
      </div>

      {/* Progress Indicator */}
      <div className="workflow-progress">
        <div className="progress-steps">
          {['Describe', 'Generate', 'Vectorize', 'Download'].map((stepName, index) => {
            // Determine if this step should be highlighted based on backend stage
            const backendStage = workflowState.currentStage || 0;
            const highlightedStage = backendStage < 3 ? backendStage : backendStage - 1;
            
            return (
              <div key={index} className={`progress-step ${index < highlightedStage ? 'completed' : ''} ${index === highlightedStage ? 'active' : ''}`}>
                <div className="step-number">{index + 1}</div>
                <div className="step-name">{stepName}</div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Workflow Stages */}
      <div className="workflow-stages">
        {/* Stage 0: Prompt */}
        <div id="stage-0" className={`workflow-stage ${workflowState.currentStage >= 0 ? 'accessible' : 'locked'} ${workflowState.currentStage === 0 ? 'active' : workflowState.currentStage > 0 ? 'completed' : ''}`}>
          <PromptSection
            isActive={workflowState.currentStage === 0}
            isCompleted={workflowState.currentStage > 0}
            prompt={workflowState.prompt}
            onComplete={(data) => handleStageComplete(0, data)}
          />
        </div>

        {/* Stage 1: Images */}
        <div id="stage-1" className={`workflow-stage ${workflowState.currentStage >= 1 ? 'accessible' : 'locked'} ${workflowState.currentStage === 1 ? 'active' : workflowState.currentStage > 1 ? 'completed' : ''}`}>
          <ImagesSection
            isActive={workflowState.currentStage === 1 || workflowState.currentStage === 2}
            isCompleted={workflowState.currentStage > 2}
            images={workflowState.images}
            prompt={workflowState.prompt}
            onComplete={(data) => handleStageComplete(2, data)}
          />
        </div>

        {/* Stage 2: Vectorize */}
        <div id="stage-2" className={`workflow-stage ${workflowState.currentStage >= 2 ? 'accessible' : 'locked'} ${workflowState.currentStage === 2 ? 'active' : workflowState.currentStage > 2 ? 'completed' : ''}`}>
          <VectorizeSection
            isActive={workflowState.currentStage === 3}
            isCompleted={workflowState.currentStage > 3}
            images={workflowState.images}
            vectorizedImages={workflowState.vectorizedImages}
            vectorizationComplete={workflowState.vectorizationComplete}
            onComplete={(data) => handleStageComplete(3, data)}
          />
        </div>

        {/* Stage 3: Font File */}
        <div id="stage-3" className={`workflow-stage ${workflowState.currentStage >= 3 ? 'accessible' : 'locked'} ${workflowState.currentStage === 3 ? 'active' : 'completed'}`}>
          <FontFileSection
            isActive={workflowState.currentStage === 4}
            isCompleted={false}
            vectorizedImages={workflowState.vectorizedImages}
            fontFileUrl={workflowState.fontFileUrl}
            onComplete={(data) => handleStageComplete(4, data)}
          />
        </div>
      </div>
    </div>
  );
}

export default WorkflowPage; 