import React, { useState, useEffect, useRef } from 'react';
import { url_base } from '../../utils';

interface VectorizeSectionProps {
  isActive: boolean;
  isCompleted: boolean;
  images: string[] | null;
  vectorizedImages: (string | null)[];
  vectorizationComplete: boolean[] | null;
  onComplete: (data: { vectorizedImages: (string | null)[] }) => void;
}

function VectorizeSection({ isActive, isCompleted, images, vectorizedImages, vectorizationComplete, onComplete }: VectorizeSectionProps) {
  const [localVectorizedImages, setLocalVectorizedImages] = useState<(string | null)[]>(vectorizedImages);
  const [localVectorizationComplete, setLocalVectorizationComplete] = useState<boolean[]>(vectorizationComplete || Array(26).fill(false));
  const [loadingStates, setLoadingStates] = useState<boolean[]>(Array(26).fill(false));
  const [progressValues, setProgressValues] = useState<number[]>(Array(26).fill(0));
  const [progressImages, setProgressImages] = useState<(string | null)[]>(Array(26).fill(null));
  const intervalRefs = useRef<(number | null)[]>(Array(26).fill(null));
  const urlExtensions = useRef<(string | null)[]>(Array(26).fill(null));

  useEffect(() => {
    // Update local state when vectorizedImages prop changes (for pre-loaded data)
    setLocalVectorizedImages(vectorizedImages);
    setLocalVectorizationComplete(vectorizationComplete || Array(26).fill(false));
  }, [vectorizedImages, vectorizationComplete]);

  useEffect(() => {
    // Clean up intervals on unmount
    return () => {
      intervalRefs.current.forEach((interval, index) => {
        if (interval !== null) {
          clearInterval(interval);
          intervalRefs.current[index] = null;
        }
      });
    };
  }, []);

  const handleVectorize = async (index: number) => {
    if (loadingStates[index] || !images || !isActive) return;
    
    console.debug(`Starting vectorization for letter ${String.fromCharCode(65 + index)}`);
    
    // Clear old vectorized image and completion status when starting new vectorization
    setLocalVectorizedImages(prevImages => {
      const newImages = [...prevImages];
      newImages[index] = null;
      return newImages;
    });
    
    setLocalVectorizationComplete(prevComplete => {
      const newComplete = [...prevComplete];
      newComplete[index] = false;
      return newComplete;
    });
    
    setLoadingStates(prevStates => {
      const newStates = [...prevStates];
      newStates[index] = true;
      return newStates;
    });
    
    setProgressValues(prevValues => {
      const newValues = [...prevValues];
      newValues[index] = 0;
      return newValues;
    });
    
    // Don't set original image as progress image - wait for actual vectorization progress
    setProgressImages(prevImages => {
      const newImages = [...prevImages];
      newImages[index] = null; // Keep empty until we get actual progress
      return newImages;
    });
    
    if (intervalRefs.current[index] !== null) {
      clearInterval(intervalRefs.current[index] as number);
      intervalRefs.current[index] = null;
    }
    
    try {
      const token = localStorage.getItem('authToken');
      const response = await fetch(`${url_base}/api/vectorization/${window.fontRunId}/${index}/sample`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ 
          image: images[index]
        })
      });
      
      if (!response.ok) {
        throw new Error(`Vectorization failed for letter ${String.fromCharCode(65 + index)}`);
      }
      
      const data = await response.json();
      console.debug(`Vectorization response for letter ${String.fromCharCode(65 + index)}:`, data);
      
      if (data && data.url_extension) {
        const newExtensions = [...urlExtensions.current];
        newExtensions[index] = data.url_extension;
        urlExtensions.current = newExtensions;
        
        console.debug(`Starting polling for letter ${String.fromCharCode(65 + index)} with extension:`, data.url_extension);
        startPolling(index);
      } else {
        throw new Error(`No URL extension returned for letter ${String.fromCharCode(65 + index)}`);
      }
    } catch (error) {
      console.error(`Error vectorizing letter ${String.fromCharCode(65 + index)}:`, error);
      setLoadingStates(prevStates => {
        const newStates = [...prevStates];
        newStates[index] = false;
        return newStates;
      });
    }
  };

  const startPolling = (index: number) => {
    console.debug(`Setting up polling interval for letter ${String.fromCharCode(65 + index)}`);
    intervalRefs.current[index] = setInterval(() => {
      checkProgress(index);
    }, 1000) as unknown as number;
  };

  const checkProgress = async (index: number) => {
    const extension = urlExtensions.current[index];
    if (!extension) {
      console.debug(`No extension found for letter ${String.fromCharCode(65 + index)}`);
      return;
    }
    
    try {
      const url = `${url_base}${extension}`;
      console.debug(`Polling for letter ${String.fromCharCode(65 + index)} at:`, url);
      
      const token = localStorage.getItem('authToken');
      const response = await fetch(url, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (!response.ok) {
        console.debug(`Polling response not OK for letter ${String.fromCharCode(65 + index)}, status:`, response.status);
        return;
      }
      
      const data = await response.json();
      console.debug(`Polling response for letter ${String.fromCharCode(65 + index)}:`, data);
      
      // CASE 1: Error response - stop polling
      if (data && data.error) {
        console.error(`Error vectorizing letter ${String.fromCharCode(65 + index)}: ${data.error}`);
        
        // Clear interval and update states
        if (intervalRefs.current[index] !== null) {
          clearInterval(intervalRefs.current[index] as number);
          intervalRefs.current[index] = null;
        }
        
        // Set loading state to false
        setLoadingStates(prevStates => {
          const newStates = [...prevStates];
          newStates[index] = false;
          return newStates;
        });
        
        // Clear any partial vectorized image so button can return to "Vectorize" state
        setLocalVectorizedImages(prevImages => {
          const newImages = [...prevImages];
          newImages[index] = null;
          return newImages;
        });
        
        // Reset progress
        setProgressValues(prevValues => {
          const newValues = [...prevValues];
          newValues[index] = 0;
          return newValues;
        });
        
        // Clear progress image
        setProgressImages(prevImages => {
          const newImages = [...prevImages];
          newImages[index] = null;
          return newImages;
        });
        
        return;
      }
      
      // CASE 4: Final image array (we're done) - stop polling
      if (Array.isArray(data) && data.length === 1 && typeof data[0] === 'string') {
        // We received the final vectorized image (base64 JPEG) - stop polling
        console.debug(`Vectorization complete for letter ${String.fromCharCode(65 + index)}, image data length:`, data[0].length);
        
        if (intervalRefs.current[index] !== null) {
          clearInterval(intervalRefs.current[index] as number);
          intervalRefs.current[index] = null;
        }
        
        // Update vectorized image (base64 JPEG)
        setLocalVectorizedImages(prevImages => {
          const newImages = [...prevImages];
          newImages[index] = data[0];
          return newImages;
        });

        setLocalVectorizationComplete(prevComplete => {
          const newComplete = [...prevComplete];
          newComplete[index] = true;
          return newComplete;
        });
        
        // Set progress to 100% for completed vectorization
        setProgressValues(prevValues => {
          const newValues = [...prevValues];
          newValues[index] = 100;
          return newValues;
        });
        
        // Update loading state
        setLoadingStates(prevStates => {
          const newStates = [...prevStates];
          newStates[index] = false;
          return newStates;
        });
        
        // Clear progress image when complete
        setProgressImages(prevImages => {
          const newImages = [...prevImages];
          newImages[index] = null;
          return newImages;
        });
        
        console.debug(`Vectorization completed for letter ${String.fromCharCode(65 + index)}`);
        return;
      }
      
      // CASE 3: Progress with image - update progress and image but continue polling
      if (data && 
         data.progress !== undefined && 
         data.image !== undefined && 
         Array.isArray(data.image) && 
         data.image.length === 1 && 
         typeof data.image[0] === 'string') {
        
        // Store the exact progress value from the API
        const progressValue = typeof data.progress === 'number' ? data.progress : 0;
        // Calculate the percentage for display (this should be exactly what the API intends)
        const percentageValue = Math.round(progressValue * 100);
        
        console.debug(`Progress update with image for letter ${String.fromCharCode(65 + index)}: ${percentageValue}%`);
        
        // Update progress value
        setProgressValues(prevValues => {
          const newValues = [...prevValues];
          newValues[index] = percentageValue;
          return newValues;
        });
        
        // Update progress image (partial result)
        setProgressImages(prevImages => {
          const newImages = [...prevImages];
          newImages[index] = data.image[0];
          return newImages;
        });
        
        // Continue polling - don't clear interval
        return;
      }
      
      // CASE 2: Just progress indicator - update progress and continue polling
      if (data && typeof data.progress === 'number') {
        // Store the exact progress value from the API
        const progressValue = data.progress;
        // Calculate the percentage for display (this should be exactly what the API intends)
        const percentageValue = Math.round(progressValue * 100);
        
        console.debug(`Progress update for letter ${String.fromCharCode(65 + index)}: ${percentageValue}%`);
        
        setProgressValues(prevValues => {
          const newValues = [...prevValues];
          newValues[index] = percentageValue;
          return newValues;
        });
        
        // Continue polling - don't clear interval
        return;
      }
      
    } catch (error) {
      console.error(`Error checking progress for letter ${String.fromCharCode(65 + index)}:`, error);
      // Don't stop polling on error, just log it
    }
  };

  const handleCancel = async (index: number) => {
    if (!loadingStates[index] || !isActive) return;
    
    console.debug(`Cancelling vectorization for letter ${String.fromCharCode(65 + index)}`);
    
    try {
      const token = localStorage.getItem('authToken');
      const response = await fetch(`${url_base}/api/vectorization/${window.fontRunId}/${index}/cancel`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (response.ok) {
        console.debug(`Successfully cancelled vectorization for letter ${String.fromCharCode(65 + index)}`);
      } else {
        console.error(`Failed to cancel vectorization for letter ${String.fromCharCode(65 + index)}`);
      }
    } catch (error) {
      console.error(`Error cancelling vectorization for letter ${String.fromCharCode(65 + index)}:`, error);
    }
    
    // Reset loading state and clear intervals regardless of API response
    if (intervalRefs.current[index] !== null) {
      clearInterval(intervalRefs.current[index] as number);
      intervalRefs.current[index] = null;
    }
    
    setLoadingStates(prevStates => {
      const newStates = [...prevStates];
      newStates[index] = false;
      return newStates;
    });
    
    setProgressValues(prevValues => {
      const newValues = [...prevValues];
      newValues[index] = 0;
      return newValues;
    });
    
    setProgressImages(prevImages => {
      const newImages = [...prevImages];
      newImages[index] = null;
      return newImages;
    });
    
    // Clear URL extension
    const newExtensions = [...urlExtensions.current];
    newExtensions[index] = null;
    urlExtensions.current = newExtensions;
  };

  const handleVectorizeAll = async () => {
    if (!images || !isActive) return;
    
    console.debug('Starting vectorization for all incomplete characters');
    
    // Loop through all characters and start vectorization for incomplete ones
    for (let i = 0; i < 26; i++) {
      // Only vectorize if not currently loading and not already completed
      if (!loadingStates[i] && !localVectorizationComplete[i]) {
        handleVectorize(i);
        // Add small delay between requests to avoid overwhelming the server
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    }
  };

  const allVectorized = () => {
    return localVectorizedImages.every(svg => svg !== null);
  };

  const handleContinue = () => {
    if (allVectorized() && isActive) {
      onComplete({ vectorizedImages: localVectorizedImages });
    }
  };

  return (
    <div className={`workflow-section vectorize-section ${isCompleted ? 'completed' : ''} ${isActive ? 'active' : ''}`}>
      <div className="section-header">
        <h2>3. Vectorize Glyphs</h2>
        {isCompleted && <div className="completion-badge">✓ Complete</div>}
      </div>
      
      <div className="section-content">
        {isCompleted ? (
          <div className="completed-content">
            <p className="completion-text">All glyphs vectorized successfully.</p>
            <div className="completed-vectors-preview">
              {localVectorizedImages.slice(0, 6).map((imageData, index) => (
                imageData && (
                  <div key={index} className="vector-preview">
                    <img 
                      src={`data:image/jpeg;base64,${imageData}`}
                      alt={`Vectorized ${String.fromCharCode(65 + index)}`}
                      width="40" 
                      height="40"
                    />
                  </div>
                )
              ))}
              {localVectorizedImages.length > 6 && <span className="more-indicator">+{localVectorizedImages.length - 6} more</span>}
            </div>
          </div>
        ) : (
          <div className="active-content">
            <p className="section-description">
              Convert bitmap images to scalable vector paths for font generation.
            </p>
            <p className="section-note">
                Note: each generation is random; if you don't like the result for a glyph, try again until you get the desired result.
            </p>
            
            {images && (
              <div className="vectorize-grid">
                <div className="font-grid" style={{ 
                  display: 'grid', 
                  gridTemplateColumns: 'repeat(5, 1fr)', 
                  gap: '20px',
                  margin: '20px 0'
                }}>
                  {images.map((base64Image, index) => (
                    <div key={index} className="letter-container" style={{
                      border: '1px solid #ddd',
                      borderRadius: '4px',
                      padding: '10px',
                      backgroundColor: '#f9f9f9'
                    }}>
                      <div className="letter-header" style={{
                        fontSize: '18px',
                        fontWeight: 'bold',
                        marginBottom: '10px',
                        textAlign: 'center'
                      }}>
                        {String.fromCharCode(65 + index)}
                      </div>
                      <div className="image-pair" style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'flex-start',
                        marginBottom: '10px',
                        gap: '10px'
                      }}>
                        {/* Original image */}
                        <div className="original-image" style={{
                          flex: '1',
                          textAlign: 'center'
                        }}>
                          <p style={{ fontSize: '12px', margin: '0 0 5px' }}>Original</p>
                          <img 
                            src={`data:image/jpeg;base64,${base64Image}`} 
                            alt={`Letter ${String.fromCharCode(65 + index)}`}
                            style={{
                              width: '100%',
                              height: 'auto',
                              border: '1px solid #ddd'
                            }}
                          />
                        </div>
                        
                        {/* Vectorized image or loading indicator */}
                        <div className="vectorized-image" style={{
                          flex: '1',
                          textAlign: 'center',
                          position: 'relative',
                          border: '0px'
                        }}>
                          <p style={{ fontSize: '12px', margin: '0 0 5px' }}>Vectorized</p>
                          
                          {/* Always show image container - either empty, partial, or completed image */}
                          <div className="image-container" style={{
                            width: '100%',
                            position: 'relative',
                            backgroundColor: '#f0f0f0',
                            borderRadius: '4px',
                            overflow: 'hidden',
                            // minHeight: localVectorizedImages[index] || progressImages[index] ? 'auto' : '120px'
                            height: 'auto'
                          }}>
                            {/* Show vectorized image if available (either partial or final) */}
                            {((localVectorizedImages[index] && localVectorizationComplete[index]) || progressImages[index]) && (
                              <img 
                                src={`data:image/jpeg;base64,${localVectorizedImages[index] || progressImages[index]}`}
                                alt={`${loadingStates[index] ? 'Partial' : 'Final'} vectorization for ${String.fromCharCode(65 + index)}`}
                                style={{
                                  width: '100%',
                                  height: 'auto',
                                  display: 'block',
                                  border: '1px solid #ddd'
                                }}
                              />
                            )}
                            
                            {/* Overlay progress bar only when loading */}
                            {loadingStates[index] && (
                              <div className="progress-overlay" style={{
                                position: 'absolute',
                                bottom: 0,
                                left: 0,
                                right: 0,
                                backgroundColor: 'rgba(0, 0, 0, 0.6)',
                                padding: '8px',
                                color: 'white',
                                textAlign: 'center'
                              }}>
                                <p style={{ margin: '0 0 5px', fontSize: '12px', color: 'white' }}>
                                  {Math.round(progressValues[index] || 0)}%
                                </p>
                                <div style={{
                                  height: '4px',
                                  width: '100%',
                                  backgroundColor: 'rgba(255, 255, 255, 0.3)',
                                  borderRadius: '2px',
                                  overflow: 'hidden'
                                }}>
                                  <div style={{
                                    height: '100%',
                                    width: `${progressValues[index] || 0}%`,
                                    backgroundColor: 'white',
                                    transition: 'width 0.3s'
                                  }}></div>
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                      
                      {/* Vectorize/Cancel button */}
                      <button 
                        onClick={() => loadingStates[index] ? handleCancel(index) : handleVectorize(index)}
                        disabled={!isActive}
                        style={{
                          width: '100%',
                          padding: '5px',
                          backgroundColor: loadingStates[index] 
                            ? '#e74c3c' // Red for cancel
                            : '#1a1a1a', // Black for vectorize
                          color: 'white',
                          border: 'none',
                          borderRadius: '4px',
                          cursor: !isActive ? 'not-allowed' : 'pointer',
                          fontSize: '12px',
                          transition: 'background-color 0.3s',
                          opacity: !isActive ? 0.6 : 1
                        }}
                      >
                        {loadingStates[index] 
                          ? 'Cancel' 
                          : 'Vectorize'
                        }
                      </button>
                    </div>
                  ))}
                </div>
                
                <div className="vectorize-controls">
                  <button 
                    className="vectorize-all-button"
                    onClick={handleVectorizeAll}
                    disabled={!isActive}
                    style={{
                      backgroundColor: '#667eea',
                      color: 'white',
                      border: 'none',
                      padding: '1rem 2rem',
                      borderRadius: '8px',
                      fontSize: '1.1rem',
                      fontWeight: '600',
                      cursor: !isActive ? 'not-allowed' : 'pointer',
                      transition: 'background 0.3s ease',
                      marginRight: '1rem',
                      opacity: !isActive ? 0.6 : 1
                    }}
                    onMouseEnter={(e) => {
                      if (isActive) (e.target as HTMLButtonElement).style.backgroundColor = '#5a6fd8';
                    }}
                    onMouseLeave={(e) => {
                      if (isActive) (e.target as HTMLButtonElement).style.backgroundColor = '#667eea';
                    }}
                  >
                    Vectorize All
                  </button>
                  <button 
                    className="continue-button"
                    onClick={handleContinue}
                    disabled={!isActive || !allVectorized()}
                  >
                    Generate Font File
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default VectorizeSection; 