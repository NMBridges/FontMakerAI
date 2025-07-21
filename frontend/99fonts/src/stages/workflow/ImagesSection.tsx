import { useState, useEffect, useRef } from 'react';
import { url_base } from '../../utils';

interface ImagesSectionProps {
  isActive: boolean;
  isCompleted: boolean;
  images: string[] | null;
  prompt: string;
  onComplete: (data: { images: string[] }) => void;
}

function ImagesSection({ isActive, isCompleted, images, prompt, onComplete }: ImagesSectionProps) {
  const [localImages, setLocalImages] = useState<string[] | null>(images);
  const [progress, setProgress] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedImages, setSelectedImages] = useState<boolean[]>(Array(26).fill(true));
  const intervalIdRef = useRef<number | null>(null);

  useEffect(() => {
    if (isActive && !localImages && window.url_extension) {
      startProgressChecking();
    }
  }, [isActive]);

  useEffect(() => {
    // Cleanup interval on unmount
    return () => {
      if (intervalIdRef.current !== null) {
        clearInterval(intervalIdRef.current);
      }
    };
  }, []);

  const startProgressChecking = () => {
    setIsLoading(true);
    
    // Clear any existing interval first
    if (intervalIdRef.current !== null) {
      clearInterval(intervalIdRef.current);
    }
    
    // Start polling every 500ms
    intervalIdRef.current = setInterval(checkProgress, 500) as unknown as number;
  };

  const checkProgress = async () => {
    try {
      if (!window.url_extension) {
        console.log('No URL extension found');
        return;
      }
      
      const token = localStorage.getItem('authToken');
      const response = await fetch(`${url_base}${window.url_extension}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (!response.ok) {
        console.log('Response not ok');
        return;
      }
      
      const data = await response.json();
      
      if (Array.isArray(data) && data.length === 26) {
        // We received our array of base64 encoded images
        if (intervalIdRef.current !== null) {
          clearInterval(intervalIdRef.current);
          intervalIdRef.current = null;
        }
        setLocalImages(data);
        setIsLoading(false);
        return;
      }
      
      if (data && data.progress) {
        setProgress(data.progress);
      }
    } catch (err) {
      console.error("Error checking progress:", err);
      setIsLoading(false);
    }
  };

  const handleRegenerate = async () => {
    if (isLoading || !isActive) return;
    
    setIsLoading(true);
    setLocalImages(null);
    setProgress(0);
    
    // Clear existing interval if it exists
    if (intervalIdRef.current !== null) {
      clearInterval(intervalIdRef.current);
      intervalIdRef.current = null;
    }
    
    try {
      const token = localStorage.getItem('authToken');
      const response = await fetch(`${url_base}/api/diffusion/${window.fontRunId}/sample`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ 
          images: selectedImages.map((selected, index) => 
            selected ? true : (localImages ? localImages[index] : true)
          ),
          prompt: prompt.trim()
        })
      });
      
      if (!response.ok) {
        throw new Error('Failed to regenerate images');
      }
      
      const data = await response.json();
      window.url_extension = data.url_extension;
      startProgressChecking();
    } catch (err: any) {
      console.log(err.message);
      setIsLoading(false);
    }
  };

  const toggleImageSelection = (index: number) => {
    if (!isActive) return;
    const newSelectedImages = [...selectedImages];
    newSelectedImages[index] = !newSelectedImages[index];
    setSelectedImages(newSelectedImages);
  };

  const handleSelectAll = () => {
    if (!isActive) return;
    setSelectedImages(Array(26).fill(true));
  };

  const handleDeselectAll = () => {
    if (!isActive) return;
    setSelectedImages(Array(26).fill(false));
  };

  const handleContinue = () => {
    if (localImages && isActive) {
      onComplete({ images: localImages });
    }
  };

  return (
    <div className={`workflow-section images-section ${isCompleted ? 'completed' : ''} ${isActive ? 'active' : ''}`}>
      <div className="section-header">
        <h2>2. Generated Glyphs</h2>
        {isCompleted && <div className="completion-badge">✓ Complete</div>}
      </div>
      
      <div className="section-content">
        {isCompleted ? (
          <div className="completed-content">
            <p className="completion-text">Bitmap glyphs generated successfully.</p>
            <div className="completed-images-preview">
              {images && images.slice(0, 6).map((base64Image, index) => (
                <img 
                  key={index}
                  src={`data:image/jpeg;base64,${base64Image}`} 
                  alt={`Letter ${String.fromCharCode(65 + index)}`} 
                  className="preview-image"
                />
              ))}
              {images && images.length > 6 && <span className="more-indicator">+{images.length - 6} more</span>}
            </div>
          </div>
        ) : (
          <div className="active-content">
            <p className="section-description">
              AI-generated bitmap images for each letter of your font.
            </p>
            
            {localImages ? (
              <div className="images-grid">
                <div className="font-grid">
                  {localImages.map((base64Image, index) => (
                    <div 
                      key={index} 
                      className={`font-grid-item ${selectedImages[index] ? 'selected' : ''}`}
                      onClick={() => toggleImageSelection(index)}
                    >
                      <div className="letter-label">{String.fromCharCode(65 + index)}</div>
                      <div className="image-container">
                        <img 
                          src={`data:image/jpeg;base64,${base64Image}`} 
                          alt={`Letter ${String.fromCharCode(65 + index)}`} 
                          className="font-image"
                        />
                        {selectedImages[index] && <div className="selection-overlay"></div>}
                      </div>
                    </div>
                  ))}
                </div>
                
                <div className="images-controls">
                  <div className="selection-controls">
                    <button onClick={handleSelectAll} disabled={!isActive}>Select All</button>
                    <button onClick={handleDeselectAll} disabled={!isActive}>Deselect All</button>
                    <button onClick={handleRegenerate} disabled={!isActive || isLoading}>
                      {isLoading ? 'Regenerating...' : 'Regenerate Selected'}
                    </button>
                  </div>
                  
                  <button 
                    className="continue-button"
                    onClick={handleContinue}
                    disabled={!isActive}
                  >
                    Continue to Vectorization
                  </button>
                </div>
              </div>
            ) : (
              <div className="loading-container">
                <p>Generating font images... {typeof progress === 'number' ? `${Math.round(progress * 100 / 32)}%` : progress}</p>
                <div className="progress-bar">
                  <div 
                    className="progress-fill" 
                    style={{ width: `${typeof progress === 'number' ? progress * 100 / 32 : 0}%` }}
                  ></div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default ImagesSection; 