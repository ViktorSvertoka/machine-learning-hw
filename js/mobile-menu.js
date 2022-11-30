(() => {
  const menuBtnRef = document.querySelector('[data-menu-button]');
  const mobileMenuRef = document.querySelector('[data-menu]');

  menuBtnRef.addEventListener('click', () => {
    const expanded =
      menuBtnRef.getAttribute('aria-expanded') === 'true' || false;

    menuBtnRef.classList.toggle('is-open');
    menuBtnRef.setAttribute('aria-expanded', !expanded);

    mobileMenuRef.classList.toggle('is-open');

    window.addEventListener('keydown', onEscKeyPress);
    function onEscKeyPress(evt) {
      if (evt.code === 'Escape') {
        menuBtnRef.classList.remove('is-open');
        mobileMenuRef.classList.remove('is-open');
      }
    }
  });
})();
