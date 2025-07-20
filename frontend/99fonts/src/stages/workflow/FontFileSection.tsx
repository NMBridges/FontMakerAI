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
  const [fontBlob, setFontBlob] = useState<Blob | null>(null);
  const [fontFaceLoaded, setFontFaceLoaded] = useState(false);

  useEffect(() => {
    if (isActive && !localFontFileUrl && vectorizedImages.every(svg => svg !== null)) {
      generateFontFile();
    }
  }, [isActive]);

  useEffect(() => {
    if (fontBlob && !fontFaceLoaded) {
      loadCustomFont();
    }
  }, [fontBlob]);

  const loadCustomFont = async () => {
    if (!fontBlob) return;
    
    console.log('Attempting to load custom font...');
    
    // Set a timeout to prevent getting stuck
    const timeoutId = setTimeout(() => {
      console.log('Font loading timeout - proceeding anyway');
      setFontFaceLoaded(true);
    }, 3000);
    
    try {
      // Create font URL from blob
      const fontUrl = URL.createObjectURL(fontBlob);
      console.log('Font URL created');
      
      // Create and load custom font face
      const fontFace = new FontFace('CustomGeneratedFont', `url(${fontUrl})`);
      
      // Try to load the font with a timeout
      await Promise.race([
        fontFace.load(),
        new Promise((_, reject) => setTimeout(() => reject(new Error('Font load timeout')), 2000))
      ]);
      
      console.log('Font loaded successfully');
      
      // Add to document fonts
      document.fonts.add(fontFace);
      
      // Clear the timeout since we succeeded
      clearTimeout(timeoutId);
      setFontFaceLoaded(true);
      
      // Clean up URL after loading
      setTimeout(() => URL.revokeObjectURL(fontUrl), 1000);
      
    } catch (error) {
      console.error('Error loading custom font:', error);
      // Clear timeout and proceed anyway
      clearTimeout(timeoutId);
      setFontFaceLoaded(true);
    }
  };

  const generateFontFile = async () => {
    if (isGenerating || !isActive) return;
    
    setIsGenerating(true);
    setError('');
    
    try {
      const token = localStorage.getItem('authToken');
      const response = await fetch(`${url_base}/api/fontrun/${window.fontRunId}/download`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (!response.ok) {
        throw new Error('Failed to generate font file');
      }
      
      // Handle the response as a file blob
      const blob = await response.blob();
      
      // Store the blob for font preview
      setFontBlob(blob);
      
      // Get filename from Content-Disposition header or use default
      const contentDisposition = response.headers.get('Content-Disposition');
      let filename = 'custom-font.otf';
      if (contentDisposition && contentDisposition.includes('filename=')) {
        const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
        if (filenameMatch && filenameMatch[1]) {
          filename = filenameMatch[1].replace(/['"]/g, '');
        }
      }
      
      // Create download URL from blob
      const downloadUrl = window.URL.createObjectURL(blob);
      
      // Trigger immediate download
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      // Clean up the URL object
      window.URL.revokeObjectURL(downloadUrl);
      
      // Set state to show completion
      setLocalFontFileUrl(downloadUrl);
      onComplete({ fontFileUrl: downloadUrl });
      
    } catch (err: any) {
      setError('Failed to generate font file. Please try again.');
      console.error('Font generation error:', err);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDownload = async () => {
    if (localFontFileUrl) {
      // If we have a blob URL, just trigger download again
      const link = document.createElement('a');
      link.href = localFontFileUrl;
      link.download = 'custom-font.otf';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } else {
      // If no local file, regenerate it
      await generateFontFile();
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
                <p style={{ 
                  fontFamily: fontFaceLoaded ? 'CustomGeneratedFont, serif' : 'serif', 
                  fontSize: '24px', 
                  textAlign: 'center', 
                  margin: '20px 0',
                  opacity: fontFaceLoaded ? 1 : 0.6,
                  transition: 'opacity 0.5s ease'
                }}>
                  ABCDEFGHIJKLMNOPQRSTUVWXYZ
                </p>
                <p style={{ 
                  fontFamily: fontFaceLoaded ? 'CustomGeneratedFont, serif' : 'serif', 
                  fontSize: '18px', 
                  textAlign: 'center', 
                  margin: '10px 0',
                  opacity: fontFaceLoaded ? 1 : 0.6,
                  transition: 'opacity 0.5s ease'
                }}>
                THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG
                </p>
                {!fontFaceLoaded && (
                  <p style={{ fontSize: '12px', color: '#666', textAlign: 'center', fontStyle: 'italic' }}>
                    Loading custom font preview...
                  </p>
                )}
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