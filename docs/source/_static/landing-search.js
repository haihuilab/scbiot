(function () {
  window.addEventListener('DOMContentLoaded', function () {
    var input = document.getElementById('scbiot-search-input');
    if (!input) {
      return;
    }

    var focusSearch = function () {
      input.focus();
      input.select();
    };

    var isFieldActive = function () {
      var active = document.activeElement;
      if (!active) {
        return false;
      }
      var tag = active.tagName ? active.tagName.toLowerCase() : '';
      return (
        active.isContentEditable ||
        tag === 'input' ||
        tag === 'textarea' ||
        tag === 'select'
      );
    };

    document.addEventListener('keydown', function (event) {
      var key = event.key ? event.key.toLowerCase() : '';
      if (key === 'k' && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        focusSearch();
        return;
      }

      if (key === '/' && !event.ctrlKey && !event.metaKey && !event.altKey) {
        if (isFieldActive()) {
          return;
        }
        event.preventDefault();
        focusSearch();
      }
    });
  });
})();
