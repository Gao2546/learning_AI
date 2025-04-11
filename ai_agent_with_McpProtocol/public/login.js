document.addEventListener('DOMContentLoaded', () => {
    const errorMessageDiv = document.getElementById('error-message');
    const urlParams = new URLSearchParams(window.location.search);
    const error = urlParams.get('error');
    const success = urlParams.get('success');

    if (error) {
        let message = 'An unexpected error occurred. Please try again.';
        if (error === 'invalid_credentials') {
            message = 'Invalid username or password.';
        } else if (error === 'server_error') {
            message = 'Server error during login. Please try again later.';
        }
        errorMessageDiv.textContent = message;
        errorMessageDiv.style.display = 'block'; // Show the error message div
        errorMessageDiv.style.color = '#ff4d4d'; // Error color
        errorMessageDiv.style.backgroundColor = '#444'; // Background for contrast
    } else if (success === 'registered') {
        errorMessageDiv.textContent = 'Registration successful! Please log in.';
        errorMessageDiv.style.display = 'block'; // Show the success message div
        errorMessageDiv.style.color = '#4dff4d'; // Success color (e.g., light green)
        errorMessageDiv.style.backgroundColor = '#444'; // Background for contrast
    }
});