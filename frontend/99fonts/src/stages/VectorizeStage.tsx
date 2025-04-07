import { useNavigate } from 'react-router-dom';

function VectorizeStage() {
  const navigate = useNavigate();
  
  return (
    <div className="stage-content">
      <h2>Vectorize Your Font</h2>
      <div className="vectorize-container">
        <p>Vectorization tools will appear here</p>
      </div>
      <button onClick={() => navigate('/font-file')}>Generate Font File</button>
    </div>
  );
}

export default VectorizeStage; 