import React, { useState } from 'react';
import { url_base } from '../../utils';

interface PromptSectionProps {
  isActive: boolean;
  isCompleted: boolean;
  prompt: string;
  onComplete: (data: { prompt: string }) => void;
}

function PromptSection({ isActive, isCompleted, prompt, onComplete }: PromptSectionProps) {
  const [localPrompt, setLocalPrompt] = useState(prompt);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const target = e.target as HTMLTextAreaElement;
    setLocalPrompt(target.value);
    target.style.height = "auto";
    target.style.height = `${target.scrollHeight}px`;
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey && isActive) {
      event.preventDefault();
      handleSubmit();
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
        body: JSON.stringify({ stage })
      });
      
      if (!response.ok) {
        console.error('Failed to update stage in backend');
      }
    } catch (error) {
      console.error('Error updating stage:', error);
    }
  };

  const handleSubmit = async () => {
    if (!localPrompt.trim()) {
      setError('Please describe your font before generating');
      return;
    }

    setLoading(true);
    setError('');
    
    try {
      const token = localStorage.getItem('authToken');
      const response = await fetch(`${url_base}/api/diffusion/${window.fontRunId}/sample`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ 
          prompt: localPrompt.trim(),
          images: Array(26).fill(true) 
        })
      });
      
      if (!response.ok) {
        throw new Error('Failed to generate font');
      }
      
      const data = await response.json();
      window.url_extension = data.url_extension;
      
      // Call onComplete to move to next stage
      onComplete({ prompt: localPrompt.trim() });
    } catch (err: any) {
      setError('Failed to generate font. Please try again.');
      console.error('Font generation error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`workflow-section prompt-section ${isCompleted ? 'completed' : ''} ${isActive ? 'active' : ''}`}>
      <div className="section-header">
        <h2>1. Describe Your Font</h2>
        {isCompleted && <div className="completion-badge">✓ Complete</div>}
      </div>
      
      <div className="section-content">
        {isCompleted ? (
          <div className="completed-content">
            <p className="completed-prompt">"{prompt}"</p>
            <p className="completion-text">Font description complete. Generating images...</p>
          </div>
        ) : (
          <div className="active-content">
            <p className="section-description">
              Describe the style and characteristics you want for your custom font.
            </p>
            <div className="prompt-input-container">
              <textarea 
                className="prompt-input" 
                placeholder="e.g., 'Modern sans-serif with rounded edges' or 'Vintage serif with decorative flourishes'"
                value={localPrompt}
                onChange={handleInput}
                onKeyDown={handleKeyDown}
                disabled={!isActive || loading}
                rows={3}
              />
              <button 
                className="generate-button"
                onClick={handleSubmit}
                disabled={!isActive || loading || !localPrompt.trim()}
              >
                {loading ? 'Generating...' : 'Generate Font'}
              </button>
            </div>
            {error && <div className="error-message">{error}</div>}
          </div>
        )}
      </div>
    </div>
  );
}

export default PromptSection; 