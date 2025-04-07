import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';

const url_base = 'http://44.210.86.218';

// Access the url_extension from the global window object
declare global {
  interface Window {
    url_extension: string;
  }
}

function ImagesStage() {
  const navigate = useNavigate();
  const [progress, setProgress] = useState(0);
  const [images, setImages] = useState<string[] | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedImages, setSelectedImages] = useState<boolean[]>(Array(26).fill(true));
  const intervalIdRef = useRef<number | null>(null);
  
  // Function to handle regeneration
  const handleRegenerate = async () => {
    if (isLoading) return; // Prevent multiple clicks
    
    setIsLoading(true);
    setImages(null);
    setProgress(0);
    
    // Clear existing interval if it exists
    if (intervalIdRef.current !== null) {
      clearInterval(intervalIdRef.current);
      intervalIdRef.current = null;
    }
    
    try {
      const data = await (await fetch(`${url_base}/api/sample_diffusion`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ 
          images: selectedImages.map((selected, index) => 
            selected ? true : (images ? images[index] : true)
          ) 
        })
      })).json();
      window.url_extension = data.url_extension;
      console.log(data);
      
      // Start a new progress checking interval
      startProgressChecking();
    } catch (err: any) {
      console.log(err.message);
      setIsLoading(false); // Make sure to reset loading state if fetch fails
    }
  };
  
  // Function to toggle image selection
  const toggleImageSelection = (index: number) => {
    const newSelectedImages = [...selectedImages];
    newSelectedImages[index] = !newSelectedImages[index];
    setSelectedImages(newSelectedImages);
  };
  
  // Function to start progress checking
  const startProgressChecking = () => {
    // Clear any existing interval first
    if (intervalIdRef.current !== null) {
      clearInterval(intervalIdRef.current);
    }
    
    // Start polling every 500ms
    intervalIdRef.current = setInterval(checkProgress, 500) as unknown as number;
  };
  
  // Function to check progress
  const checkProgress = async () => {
    try {
      if (!window.url_extension) {
        console.log('No URL extension found');
        return;
      }
      const response = await fetch(`${url_base}${window.url_extension}`);
      if (!response.ok) {
        console.log('Response not ok');
        return;
      }
      
      const data = await response.json();
      console.log(data);
      
      if (Array.isArray(data) && data.length === 26) {
        // We received our array of base64 encoded images
        if (intervalIdRef.current !== null) {
          clearInterval(intervalIdRef.current);
          intervalIdRef.current = null;
        }
        setImages(data);
        setIsLoading(false);
        return;
      }
      
      if (data && data.progress) {
        setProgress(data.progress);
      }
    } catch (err) {
      console.error("Error checking progress:", err);
      setIsLoading(false); // Reset loading state on error
    }
  };
  
  useEffect(() => {
    // Start checking progress when component mounts
    if (window.url_extension) {
      startProgressChecking();
    }
    
    // Clean up interval on component unmount
    return () => {
      if (intervalIdRef.current !== null) {
        clearInterval(intervalIdRef.current);
      }
    };
  }, []);
  
  // Set loading state when component mounts if no images are available
  useEffect(() => {
    if (!images && window.url_extension) {
      setIsLoading(true);
    }
  }, []);
  
  return (
    <div className="stage-content">
      <h2>Glyphs</h2>
      <div className="images-container">
        {images ? (
          <div className="font-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(10, 1fr)', gap: '10px' }}>
            {images.map((base64Image, index) => (
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
      <div className="button-container">
        <button 
          onClick={handleRegenerate}
          disabled={!images || isLoading}
          className="regenerate-button"
        >
          {isLoading ? 'Generating...' : 'Regenerate Selected'}
        </button>
        <button 
          onClick={() => navigate('/vectorize')}
          disabled={!images || isLoading}
          className="convert-button"
        >
          Convert to Vector Paths
        </button>
      </div>
    </div>
  );
}

export default ImagesStage; 