document.addEventListener('DOMContentLoaded', () => {
  const transitionElement = document.createElement('div');
  transitionElement.className = 'page-transition';
  
  const logoAnimation = document.createElement('div');
  logoAnimation.className = 'logo-animation';
  
  document.body.appendChild(transitionElement);
  document.body.appendChild(logoAnimation);

  const mainContent = document.querySelector('.min-h-screen.flex.items-center.justify-center');
  if (mainContent) {
    mainContent.classList.add('content-container');
  }

  const links = document.querySelectorAll('a[href="get-Started.html"], a[href="index.html"], a.bg-green-600, .absolute.top-6.left-6 a');
  
  links.forEach(link => {
    link.addEventListener('click', function(e) {
      console.log('Link clicked:', this.getAttribute('href'));
      e.preventDefault();
      
      const href = this.getAttribute('href');
      const currentPage = window.location.pathname.split('/').pop() || 'index.html';
      const targetPage = href.startsWith('#') ? 'get-Started.html' : href;
      
      if (currentPage === targetPage) return;
      
      const content = document.querySelector('.content-container') || document.body;
      
      content.classList.add('fade-out');
      
      setTimeout(() => {
        transitionElement.classList.add('active');
        
        setTimeout(() => {
          logoAnimation.classList.add('active');
          
          logoAnimation.style.animation = 'spin 1s ease-in-out';
          
          setTimeout(() => {
            window.location.href = targetPage;
          }, 500);
        }, 200);
      }, 200);
    });
  });
  

  const goBackButton = document.querySelector('.absolute.top-6.left-6 a');
  if (goBackButton) {
    console.log('Go Back button found:', goBackButton);
    goBackButton.addEventListener('click', function(e) {
      console.log('Go Back button clicked');
      e.preventDefault();
      
      const content = document.querySelector('.content-container') || document.body;
      content.classList.add('fade-out');
      
      setTimeout(() => {
        transitionElement.classList.add('active');
        
        setTimeout(() => {
          logoAnimation.classList.add('active');
          logoAnimation.style.animation = 'spin 1s ease-in-out';
          
          setTimeout(() => {
            window.location.href = 'index.html';
          }, 500);
        }, 200);
      }, 200);
    });
  }
  
  window.addEventListener('load', function() {
    const transitionElement = document.querySelector('.page-transition');
    const logoAnimation = document.querySelector('.logo-animation');
    
    if (transitionElement && logoAnimation) {
      transitionElement.classList.add('active');
      logoAnimation.classList.add('active');
      
      setTimeout(() => {
        logoAnimation.classList.remove('active');
        
        setTimeout(() => {
          transitionElement.classList.remove('active');
          
          const mainContent = document.querySelector('.flex.items-center.justify-center');
          if (mainContent && !mainContent.classList.contains('content-container')) {
            mainContent.classList.add('content-container');
          }
        }, 400);
      }, 600);
    }
  });
  
  window.addEventListener('pageshow', function(event) {
    if (event.persisted) {
      location.reload();
    }
  });
});