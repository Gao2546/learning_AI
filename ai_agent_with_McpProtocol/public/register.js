document.addEventListener('DOMContentLoaded', () => {
    const errorMessageDiv = document.getElementById('error-message');
    const urlParams = new URLSearchParams(window.location.search);
    const error = urlParams.get('error');

    if (error) {
        let message = 'An unexpected error occurred. Please try again.';
        if (error === 'username_exists') {
            message = 'Username already exists. Please choose another one.';
        } else if (error === 'email_exists') {
            message = 'Email already registered. Please use a different email or log in.';
        } else if (error === 'server_error') {
            message = 'Server error during registration. Please try again later.';
        }
        errorMessageDiv.textContent = message;
        errorMessageDiv.style.display = 'block'; // Show the error message div
    }
});