import { useState } from 'react'
import fontsLogo from '/99fonts.svg'
import './App.css'

function App() {
  const [count, setCount] = useState(0)

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
        <button className="generate-button">Generate</button>
      </div>
      <p className="read-the-docs">
        Contact glyphpy@gmail.com to learn more.
      </p>
    </>
  )
}

export default App
