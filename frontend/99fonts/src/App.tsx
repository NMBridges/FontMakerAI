import { useState } from 'react'
import fontsLogo from '/99fonts.svg'
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  const handleClick = async () => {
    try {
        const data = await (await fetch('http://localhost:8080/sample_diffusion', {headers: {'Content-Type': 'application/json'}})).json();
        console.log(data);
    } catch (err: any) {
        console.log(err.message);
    }
  }

  return (
    <>
      <div className="logo-container">
        <a target="_blank">
          <img src={fontsLogo} className="logo react" alt="99 Fonts logo" />
        </a>
      </div>
      <h1>Copyright-free, custom fonts. Powered by AI.</h1>
      <div className="search-bar-container">
        <textarea className="search-bar" placeholder="Describe your next font"></textarea>
        <button className="generate-button" onClick={handleClick}>Generate</button>
      </div>
      <p className="read-the-docs">
        Contact glyphpy@gmail.com to learn more.
      </p>
    </>
  )
}

export default App
