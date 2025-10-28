import { BrowserRouter, Routes, Route } from 'react-router-dom'
import HomePage from './HomePage'
import UFCPage from './pages/ufc/UFCPage'
import About from './pages/about/About'
// import { Captcha } from './Captcha'

function App() {
  return (
    // <Captcha>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/about" element={<About />} />
          <Route path="/ufc" element={<UFCPage />} />
        </Routes>
      </BrowserRouter>
    // </Captcha>
  )
}

export default App