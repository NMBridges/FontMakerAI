import { useNavigate } from 'react-router-dom';
import { useState, useEffect, useRef } from 'react';
import { url_base } from '../utils';

// Access the url_extension from the global window object
declare global {
  interface Window {
    url_extension: string;
  }
}

function VectorizeStage() {
  const navigate = useNavigate();
  const [images, setImages] = useState<string[] | null>(null);
  const [vectorizedImages, setVectorizedImages] = useState<(string | null)[]>(Array(26).fill(null));
  const [loadingStates, setLoadingStates] = useState<boolean[]>(Array(26).fill(false));
  const [progressValues, setProgressValues] = useState<number[]>(Array(26).fill(0));
  const [progressPercentages, setProgressPercentages] = useState<number[]>(Array(26).fill(0));
  const intervalRefs = useRef<(number | null)[]>(Array(26).fill(null));
  const urlExtensions = useRef<(string | null)[]>(Array(26).fill(null));
  
  // Load images when component mounts
  useEffect(() => {
    const fetchImages = async () => {
      if (!window.url_extension) {
        console.error('No URL extension found. Please generate images first.');
        return;
      }
      
      try {
        const response = await fetch(`${url_base}${window.url_extension}`);
        if (!response.ok) {
          throw new Error('Failed to fetch images');
        }
        
        const data = await response.json();
        if (Array.isArray(data) && data.length > 0) {
          setImages(data);
        } else {
          console.error('Invalid image data format');
        }
      } catch (err) {
        console.error("Error fetching images:", err);
      }
    };
    
    fetchImages();
    
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
  
  // Function to handle vectorization for a specific letter
  const handleVectorize = async (index: number) => {
    if (loadingStates[index] || !images) return;
    
    // Update loading state for this specific letter using functional update
    setLoadingStates(prevStates => {
      const newStates = [...prevStates];
      newStates[index] = true;
      return newStates;
    });
    
    // Reset progress for this letter using functional update
    setProgressValues(prevValues => {
      const newValues = [...prevValues];
      newValues[index] = 0;
      return newValues;
    });
    
    // Clear existing interval if any
    if (intervalRefs.current[index] !== null) {
      clearInterval(intervalRefs.current[index] as number);
      intervalRefs.current[index] = null;
    }
    
    try {
      const response = await fetch(`${url_base}/api/sample_path`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ image: images[index] })
      });
      
      if (!response.ok) {
        throw new Error(`Vectorization failed for letter ${String.fromCharCode(65 + index)}`);
      }
      
      const data = await response.json();
      console.log(`Vectorization initiated for letter ${String.fromCharCode(65 + index)}:`, data);
      
      if (data && data.url_extension) {
        // Create a new reference for this specific index only
        const newExtensions = [...urlExtensions.current];
        newExtensions[index] = data.url_extension;
        urlExtensions.current = newExtensions;
        
        // Start polling for progress
        startPolling(index);
      } else {
        throw new Error(`No URL extension returned for letter ${String.fromCharCode(65 + index)}`);
      }
    } catch (err: any) {
      console.error(`Error during vectorization for letter ${String.fromCharCode(65 + index)}:`, err.message);
      
      // Update loading state using functional update
      setLoadingStates(prevStates => {
        const newStates = [...prevStates];
        newStates[index] = false;
        return newStates;
      });
    }
  };
  
  // Function to cancel vectorization for a specific letter
  const handleCancelVectorization = async (index: number) => {
    try {
      const urlExtension = urlExtensions.current[index];
      
      if (!urlExtension) {
        console.error(`No URL extension found for letter ${String.fromCharCode(65 + index)}`);
        return;
      }
      
      // First, clear the interval to stop polling immediately
      if (intervalRefs.current[index] !== null) {
        clearInterval(intervalRefs.current[index] as number);
        console.log(`Cleared interval for letter ${String.fromCharCode(65 + index)}`);
        
        const newIntervals = [...intervalRefs.current];
        newIntervals[index] = null;
        intervalRefs.current = newIntervals;
      }
      
      // Send cancel request to the API
      const response = await fetch(`${url_base}${urlExtension}/cancel`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'}
      });
      
      console.log(`Cancel request sent for letter ${String.fromCharCode(65 + index)}`);
      
      // Reset loading state
      setLoadingStates(prevStates => {
        const newStates = [...prevStates];
        newStates[index] = false;
        return newStates;
      });
      
      // Clear partial vectorized image if exists
      if (vectorizedImages[index]) {
        setVectorizedImages(prevImages => {
          const newImages = [...prevImages];
          newImages[index] = null;
          return newImages;
        });
      }
      
      // Reset progress
      setProgressValues(prevValues => {
        const newValues = [...prevValues];
        newValues[index] = 0;
        return newValues;
      });
      
      setProgressPercentages(prevPercentages => {
        const newPercentages = [...prevPercentages];
        newPercentages[index] = 0;
        return newPercentages;
      });
      
    } catch (err) {
      console.error(`Error cancelling vectorization for letter ${String.fromCharCode(65 + index)}:`, err);
      
      // Even if cancel request fails, ensure interval is cleared and UI is reset
      if (intervalRefs.current[index] !== null) {
        clearInterval(intervalRefs.current[index] as number);
        
        const newIntervals = [...intervalRefs.current];
        newIntervals[index] = null;
        intervalRefs.current = newIntervals;
      }
      
      // Reset loading state
      setLoadingStates(prevStates => {
        const newStates = [...prevStates];
        newStates[index] = false;
        return newStates;
      });
    }
  };
  
  // Function to start polling for progress updates
  const startPolling = (index: number) => {
    // Clear existing interval if any
    if (intervalRefs.current[index] !== null) {
      clearInterval(intervalRefs.current[index] as number);
    }
    
    // Create a fresh reference for this interval ID
    const newIntervals = [...intervalRefs.current];
    newIntervals[index] = setInterval(() => checkProgress(index), 500) as unknown as number;
    intervalRefs.current = newIntervals;
  };
  
  // Function to check progress for a specific letter
  const checkProgress = async (index: number) => {
    try {
      // Get the URL extension for this specific index only
      const urlExtension = urlExtensions.current[index];
      
      if (!urlExtension) {
        console.log(`No URL extension found for letter ${String.fromCharCode(65 + index)}`);
        return;
      }
      
      const response = await fetch(`${url_base}${urlExtension}`);
      if (!response.ok) {
        console.log(`Response not ok for letter ${String.fromCharCode(65 + index)}`);
        return;
      }
      
      const data = await response.json();
      console.log(`Progress update for letter ${String.fromCharCode(65 + index)}:`, data);
      
      // CASE 1: Error response - stop polling
      if (data && data.error) {
        console.error(`Error vectorizing letter ${String.fromCharCode(65 + index)}: ${data.error}`);
        
        // Clear interval and update states
        if (intervalRefs.current[index] !== null) {
          clearInterval(intervalRefs.current[index] as number);
          
          const newIntervals = [...intervalRefs.current];
          newIntervals[index] = null;
          intervalRefs.current = newIntervals;
        }
        
        // Set loading state to false
        setLoadingStates(prevStates => {
          const newStates = [...prevStates];
          newStates[index] = false;
          return newStates;
        });
        
        // Clear any partial vectorized image so button can return to "Vectorize" state
        setVectorizedImages(prevImages => {
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
        
        setProgressPercentages(prevPercentages => {
          const newPercentages = [...prevPercentages];
          newPercentages[index] = 0;
          return newPercentages;
        });
        
        return;
      }
      
      // CASE 4: Final image array (we're done) - stop polling
      if (Array.isArray(data) && data.length === 1 && typeof data[0] === 'string') {
        // We received the final vectorized image - stop polling
        if (intervalRefs.current[index] !== null) {
          clearInterval(intervalRefs.current[index] as number);
          
          const newIntervals = [...intervalRefs.current];
          newIntervals[index] = null;
          intervalRefs.current = newIntervals;
        }
        
        // Update vectorized image
        setVectorizedImages(prevImages => {
          const newImages = [...prevImages];
          newImages[index] = data[0];
          return newImages;
        });
        
        // Set progress to 100% for completed vectorization
        setProgressValues(prevValues => {
          const newValues = [...prevValues];
          newValues[index] = 1.0; // 100%
          return newValues;
        });
        
        // Set progress percentage to 100 for completed vectorization
        setProgressPercentages(prevPercentages => {
          const newPercentages = [...prevPercentages];
          newPercentages[index] = 100;
          return newPercentages;
        });
        
        // Update loading state
        setLoadingStates(prevStates => {
          const newStates = [...prevStates];
          newStates[index] = false;
          return newStates;
        });
        
        console.log(`Vectorization completed for letter ${String.fromCharCode(65 + index)}`);
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
        
        // Update progress value
        setProgressValues(prevValues => {
          const newValues = [...prevValues];
          newValues[index] = progressValue;
          return newValues;
        });
        
        // Update progress percentage for display
        setProgressPercentages(prevPercentages => {
          const newPercentages = [...prevPercentages];
          newPercentages[index] = percentageValue;
          return newPercentages;
        });
        
        // Update vectorized image (partial result)
        setVectorizedImages(prevImages => {
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
        
        setProgressValues(prevValues => {
          const newValues = [...prevValues];
          newValues[index] = progressValue;
          return newValues;
        });
        
        // Update progress percentage for display
        setProgressPercentages(prevPercentages => {
          const newPercentages = [...prevPercentages];
          newPercentages[index] = percentageValue;
          return newPercentages;
        });
        
        // Continue polling - don't clear interval
        return;
      }
      
    } catch (err) {
      console.error(`Error checking progress for letter ${String.fromCharCode(65 + index)}:`, err);
      // Don't stop polling on error, just log it
    }
  };
  
  // Function to check if all letters have been vectorized
  const allVectorized = () => {
    return vectorizedImages.every((img, idx) => img !== null || images?.[idx] === undefined);
  };
  
  return (
    <div className="stage-content">
      <h2>Vectorize Your Font</h2>
      <div className="vectorize-container">
        {!images ? (
          <p>Loading images...</p>
        ) : (
          <div>
            <p>Click the "Vectorize" button next to each letter to convert it to vector paths.</p>
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
                    alignItems: 'center',
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
                      position: 'relative'
                    }}>
                      <p style={{ fontSize: '12px', margin: '0 0 5px' }}>Vectorized</p>
                      
                      {/* Always show image container - either empty, partial, or completed image */}
                      <div className="image-container" style={{
                        width: '100%',
                        position: 'relative',
                        border: '1px solid #ddd',
                        backgroundColor: '#f0f0f0',
                        paddingBottom: vectorizedImages[index] ? '0' : '100%',
                        borderRadius: '4px',
                        overflow: 'hidden'
                      }}>
                        {/* Show vectorized image if available (either partial or final) */}
                        {vectorizedImages[index] && (
                          <img 
                            src={`data:image/jpeg;base64,${vectorizedImages[index]}`}
                            alt={`${loadingStates[index] ? 'Partial' : 'Final'} vectorization for ${String.fromCharCode(65 + index)}`}
                            style={{
                              width: '100%',
                              height: 'auto',
                              display: 'block'
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
                              {progressPercentages[index]}%
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
                                width: `${progressValues[index] * 100}%`,
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
                    onClick={() => loadingStates[index] 
                      ? handleCancelVectorization(index) 
                      : handleVectorize(index)
                    }
                    disabled={false}
                    style={{
                      width: '100%',
                      padding: '5px',
                      backgroundColor: loadingStates[index] 
                        ? '#e74c3c' // Red for cancel
                        : '#1a1a1a', // Black for vectorize
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      fontSize: '12px',
                      transition: 'background-color 0.3s'
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
          </div>
        )}
      </div>
      
      <div className="button-container" style={{ 
        marginTop: '20px',
        display: 'flex',
        justifyContent: 'space-between'
      }}>
        <button 
          onClick={() => navigate('/images')}
          className="navigate-button"
          style={{
            padding: '10px 20px',
            backgroundColor: '#f0f0f0',
            border: '1px solid #ddd',
            borderRadius: '4px'
          }}
        >
          Back to Glyphs
        </button>
        
        <button 
          onClick={() => navigate('/font-file')}
          disabled={!allVectorized()}
          className="navigate-button"
          style={{ 
            padding: '10px 20px',
            backgroundColor: allVectorized() ? '#1a1a1a' : '#cccccc',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: allVectorized() ? 'pointer' : 'not-allowed'
          }}
        >
          Generate Font File
        </button>
      </div>
    </div>
  );
}

export default VectorizeStage; 