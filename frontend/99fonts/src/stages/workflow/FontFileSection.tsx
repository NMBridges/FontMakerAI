import React, { useState, useEffect } from 'react';
import { url_base } from '../../utils';

interface FontFileSectionProps {
  isActive: boolean;
  isCompleted: boolean;
  vectorizedImages: (string | null)[];
  fontFileUrl: string | null;
  onComplete: (data: { fontFileUrl: string }) => void;
}

function FontFileSection({ isActive, isCompleted, vectorizedImages, fontFileUrl, onComplete }: FontFileSectionProps) {
  const [localFontFileUrl, setLocalFontFileUrl] = useState<string | null>(fontFileUrl);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (isActive && !localFontFileUrl && vectorizedImages.every(svg => svg !== null)) {
      generateFontFile();
    }
  }, [isActive]);

  const generateFontFile = async () => {
    if (isGenerating || !isActive) return;
    
    setIsGenerating(true);
    setError('');
    
    try {
      const token = localStorage.getItem('authToken');
      const response = await fetch(`${url_base}/api/generate-font`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          vectorPaths: vectorizedImages
        })
      });
      
      if (!response.ok) {
        throw new Error('Failed to generate font file');
      }
      
      const data = await response.json();
      
      if (data.fontUrl) {
        setLocalFontFileUrl(data.fontUrl);
        onComplete({ fontFileUrl: data.fontUrl });
      } else {
        throw new Error('No font URL returned from server');
      }
    } catch (err: any) {
      setError('Failed to generate font file. Please try again.');
      console.error('Font generation error:', err);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDownload = () => {
    if (localFontFileUrl) {
      const link = document.createElement('a');
      link.href = localFontFileUrl;
      link.download = 'custom-font.otf';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  return (
    <div className={`workflow-section font-file-section ${isCompleted ? 'completed' : ''} ${isActive ? 'active' : ''}`}>
      <div className="section-header">
        <h2>4. Download Your Font</h2>
        {localFontFileUrl && <div className="completion-badge">✓ Complete</div>}
      </div>
      
      <div className="section-content">
        {localFontFileUrl ? (
          <div className="completed-content">
            <div className="font-ready">
              <div className="success-icon">🎉</div>
              <h3>Your Font is Ready!</h3>
              <p className="success-message">
                Your custom font has been generated successfully and is ready for download.
              </p>
              
              <div className="font-preview">
                <p style={{ fontFamily: 'serif', fontSize: '24px', textAlign: 'center', margin: '20px 0' }}>
                  ABCDEFGHIJKLMNOPQRSTUVWXYZ
                </p>
                <p style={{ fontFamily: 'serif', fontSize: '18px', textAlign: 'center', margin: '10px 0' }}>
                  The quick brown fox jumps over the lazy dog
                </p>
              </div>
              
              <div className="download-section">
                <button 
                  className="download-button"
                  onClick={handleDownload}
                >
                  Download Font File (.otf)
                </button>
                
                <div className="usage-instructions">
                  <h4>How to use your font:</h4>
                  <ol>
                    <li>Download the font file</li>
                    <li>Install it on your system by double-clicking</li>
                    <li>Use it in any application that supports custom fonts</li>
                  </ol>
                </div>
              </div>
            </div>
          </div>
        ) : isGenerating ? (
          <div className="active-content">
            <div className="generating-font">
              <div className="loading-spinner">⏳</div>
              <h3>Generating Your Font File</h3>
              <p>Please wait while we create your custom font file from the vectorized glyphs...</p>
            </div>
          </div>
        ) : (
          <div className="active-content">
            <div className="waiting-for-vectors">
              <h3>Ready to Generate Font</h3>
              <p>Complete the vectorization step above to generate your font file.</p>
              
              {error && (
                <div className="error-section">
                  <p className="error-message">{error}</p>
                  <button 
                    className="retry-button"
                    onClick={generateFontFile}
                    disabled={!isActive}
                  >
                    Try Again
                  </button>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default FontFileSection; 