import { useNavigate } from 'react-router-dom';
import { url_base } from '../utils';

// Declare the url_extension variable
declare global {
  interface Window {
    url_extension: string;
  }
}

// Initialize the global variable if it doesn't exist
if (typeof window.url_extension === 'undefined') {
  window.url_extension = '';
}

function PromptStage() {
  const navigate = useNavigate();

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      handleClick();
    }
  };
  
  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const target = e.target as HTMLTextAreaElement;
    target.style.height = "auto";
    target.style.height = `${target.scrollHeight}px`;
  }

  const handleClick = async () => {
    try {
      navigate('/images');
      const data = await (await fetch(`${url_base}/api/sample_diffusion`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ images: Array(26).fill(true) })
      })).json();
      window.url_extension = data.url_extension;
      console.log(data);
    } catch (err: any) {
      console.log(err.message);
    }
  }

  return (
    <div className="stage-content">
      <h1 style={{marginTop: '3em'}}>Copyright-free, custom fonts. Powered by AI.</h1>
      <div className="search-bar-container">
        <textarea className="search-bar" placeholder="Describe your next font" onInput={handleInput} onKeyDown={handleKeyDown}></textarea>
        <button className="generate-button" onClick={handleClick}>Generate</button> 
      </div>
    </div>
  );
}

export default PromptStage; 