    async function loadPage(page) {

    const response = await fetch(`/pages/${page}`);

    const html = await response.text();

    document.querySelector('.main-content').innerHTML = html;
}