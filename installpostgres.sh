#!/bin/bash

# --- Script Setup ---
set -e
# set -u
set -o pipefail

echo "--- PostgreSQL User Setup Script Started ---"

# --- Determine Package Manager & Sudo ---
PKG_MANAGER=""
INSTALL_CMD=""
UPDATE_CMD=""
CHECK_PKG_CMD=""
INSTALL_OPTS="-y"
SUDO_CMD=""
NEEDS_UPDATE=false

echo "Detecting package manager and checking for sudo..."

# Check for sudo first
if [[ $EUID -ne 0 ]]; then
    if command -v sudo &> /dev/null; then
        SUDO_CMD="sudo"
        echo "Using sudo for privileged operations."
        $SUDO_CMD -v
    else
        echo "ERROR: Running as non-root and 'sudo' command not found." >&2
        exit 1
    fi
else
    echo "Running as root."
fi

# Detect package manager
if command -v apt-get &> /dev/null; then
    PKG_MANAGER="apt"
    UPDATE_CMD="apt-get update"
    INSTALL_CMD="apt-get install"
    # Assuming standard package names for Debian/Ubuntu
    PG_SERVER_PKG="postgresql"
    PG_CLIENT_PKG="postgresql-client"
    PG_SERVICE_NAME="postgresql" # systemd service name
    NEEDS_UPDATE=true
    echo "Using apt-get (Debian/Ubuntu)."
elif command -v dnf &> /dev/null; then
    PKG_MANAGER="dnf"
    UPDATE_CMD="dnf check-update"
    INSTALL_CMD="dnf install"
    # Assuming standard package names for Fedora/RHEL Stream
    PG_SERVER_PKG="postgresql-server"
    PG_CLIENT_PKG="postgresql" # Client tools often in main postgresql pkg
    PG_SERVICE_NAME="postgresql" # systemd service name
    echo "Using dnf (Fedora/RHEL/CentOS Stream)."
elif command -v yum &> /dev/null; then
    PKG_MANAGER="yum"
    UPDATE_CMD="yum check-update"
    INSTALL_CMD="yum install"
    # Assuming standard package names for older RHEL/CentOS
    PG_SERVER_PKG="postgresql-server"
    PG_CLIENT_PKG="postgresql"
    PG_SERVICE_NAME="postgresql"
    echo "Using yum (Older RHEL/CentOS)."
else
    echo "Error: Could not find a supported package manager (apt-get, dnf, or yum)." >&2
    exit 1
fi

# --- Helper Function: Run command with sudo if needed ---
run_cmd() {
    local cmd_string="$*"
    echo "Running: ${SUDO_CMD} ${cmd_string}"
    if ! eval "${SUDO_CMD} ${cmd_string}"; then
        echo "Error: Command failed: ${SUDO_CMD} ${cmd_string}" >&2
        return 1
    fi
    return 0
}

# --- Helper Function: Update package lists if needed ---
update_package_lists_if_needed() {
    if [[ "$NEEDS_UPDATE" == "true" ]]; then
        echo "Running package list update ($UPDATE_CMD)..."
        if ! run_cmd "$UPDATE_CMD"; then
            echo "Error: Failed to update package lists." >&2
            exit 1
        fi
        NEEDS_UPDATE=false
    fi
}

# --- Helper Function: Ensure PostgreSQL is installed ---
# Installs both server and client if 'psql' command is missing
ensure_postgresql_installed() {
    echo "Checking for psql command (package '$PG_CLIENT_PKG')..."
    if ! command -v psql &> /dev/null; then
        echo "'psql' command not found. Attempting to install PostgreSQL server and client..."
        update_package_lists_if_needed || exit 1

        # Install both server and client packages
        echo "Installing $PG_SERVER_PKG and $PG_CLIENT_PKG..."
        if ! run_cmd "${INSTALL_CMD} ${INSTALL_OPTS} ${PG_SERVER_PKG} ${PG_CLIENT_PKG}"; then
            echo "ERROR: Failed to install PostgreSQL packages." >&2
            exit 1
        fi

        # Special handling for RHEL/CentOS/Fedora: Initialize DB and enable/start service
        if [[ "$PKG_MANAGER" == "dnf" || "$PKG_MANAGER" == "yum" ]]; then
            # Check if data directory is initialized (e.g., /var/lib/pgsql/data/PG_VERSION)
            # This check is basic; a more robust check might look for specific files.
             PG_DATA_DIR=$(sudo -u postgres PGPASSWORD= psql -tAc "SHOW data_directory;") || PG_DATA_DIR="/var/lib/pgsql/data" # Default guess if query fails early
             if [ ! -f "${PG_DATA_DIR}/postgresql.conf" ]; then
                echo "PostgreSQL data directory seems uninitialized. Attempting initialization..."
                # Find the setup binary (path might vary slightly)
                PG_SETUP_BIN=$(find /usr/ -name "postgresql*-setup" | head -n 1)
                if [[ -n "$PG_SETUP_BIN" ]] && [[ -x "$PG_SETUP_BIN" ]]; then
                     if ! run_cmd "$PG_SETUP_BIN --initdb"; then
                         echo "ERROR: Failed to initialize PostgreSQL database." >&2
                         exit 1
                     fi
                     echo "PostgreSQL database initialized."
                else
                    echo "WARN: Could not find postgresql*-setup binary to initialize DB automatically." >&2
                     # Attempt standard initdb if available
                     if command -v initdb &> /dev/null; then
                        if ! run_cmd "initdb -D $PG_DATA_DIR"; then # May require running as postgres user
                           echo "ERROR: initdb failed." >&2
                           # exit 1 # Might continue if service can start anyway
                        fi
                     else
                        echo "WARN: Cannot automatically initialize PostgreSQL DB." >&2
                     fi
                fi
            fi
             echo "Ensuring PostgreSQL service (${PG_SERVICE_NAME}) is enabled and started..."
             run_cmd "systemctl enable ${PG_SERVICE_NAME}" || echo "WARN: Failed to enable service."
             run_cmd "systemctl start ${PG_SERVICE_NAME}" || echo "WARN: Failed to start service."
        fi

        # Verify psql again
        if ! command -v psql &> /dev/null; then
             echo "ERROR: PostgreSQL packages installed, but 'psql' command still not found." >&2
             exit 1
        fi
        echo "PostgreSQL installed successfully."

        # Wait briefly for service to become active
        echo "Waiting briefly for the PostgreSQL service..."
        sleep 5 # Simple wait
        # Robust check (requires systemctl)
        if command -v systemctl &> /dev/null; then
            echo "Checking service status (${PG_SERVICE_NAME})..."
            if ! run_cmd "systemctl is-active --quiet ${PG_SERVICE_NAME}"; then
                 echo "WARN: PostgreSQL service (${PG_SERVICE_NAME}) does not appear to be active after installation." >&2
                 echo "Attempting to start it..."
                 run_cmd "systemctl start ${PG_SERVICE_NAME}" || echo "WARN: Failed to start service again."
                 sleep 2
                 run_cmd "systemctl status ${PG_SERVICE_NAME}" # Show status regardless
            else
                echo "PostgreSQL service is active."
            fi
        fi

    else
        echo "'psql' command found. Assuming PostgreSQL server is installed and running."
        # Optional: Check service status even if psql exists
        if command -v systemctl &> /dev/null; then
             if ! run_cmd "systemctl is-active --quiet ${PG_SERVICE_NAME}"; then
                  echo "WARN: 'psql' found, but service '${PG_SERVICE_NAME}' is not active. Manual check needed." >&2
             fi
        fi
    fi
}

# --- Ensure PostgreSQL is Installed ---
ensure_postgresql_installed

# --- Get Database User Credentials ---
DEFAULT_DB_USER="athip"
read -p "Enter PostgreSQL username to create [${DEFAULT_DB_USER}]: " DB_USER
DB_USER="${DB_USER:-$DEFAULT_DB_USER}" # Use default if empty

# Prompt for password securely
while true; do
    read -sp "Enter password for user '${DB_USER}': " DB_PASSWORD
    echo # Newline after prompt
    read -sp "Confirm password: " DB_PASSWORD_CONFIRM
    echo # Newline after prompt
    if [[ "$DB_PASSWORD" == "$DB_PASSWORD_CONFIRM" ]]; then
        if [[ -z "$DB_PASSWORD" ]]; then
            echo "Password cannot be empty. Please try again."
        else
            break # Passwords match and are not empty
        fi
    else
        echo "Passwords do not match. Please try again."
    fi
done


# --- Create the PostgreSQL user ---
echo "Attempting to create PostgreSQL user '${DB_USER}'..."

# Check if the user already exists first to make the script idempotent
# Need to run psql as the 'postgres' superuser
# Use PGPASSWORD environment variable (empty for peer auth) or rely on pg_hba.conf
# Adding -X avoids reading .psqlrc which might interfere
USER_EXISTS=$( $SUDO_CMD -u postgres PGPASSWORD= psql -X -tAc "SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}'" )

if [ "$USER_EXISTS" = "1" ]; then
    echo "PostgreSQL user '${DB_USER}' already exists. Skipping creation."
    # Optional: Offer to update password?
else
    echo "User '${DB_USER}' does not exist. Creating..."
    # Escape single quotes within the password if the password itself contains them
    # Since we read the password, it shouldn't have shell metacharacters issues here,
    # but using SQL parameters via a client library is safer in general applications.
    # For psql -c, doubling single quotes inside the SQL string is the standard way.
    SQL_SAFE_PASSWORD=$(echo "$DB_PASSWORD" | sed "s/'/''/g")

    # Execute the CREATE USER command
    CREATE_SQL="CREATE USER \"${DB_USER}\" WITH PASSWORD '${SQL_SAFE_PASSWORD}';"
    if $SUDO_CMD -u postgres PGPASSWORD= psql -X -c "$CREATE_SQL"; then
        echo "PostgreSQL user '${DB_USER}' created successfully."
    else
        echo "ERROR: Failed to create PostgreSQL user '${DB_USER}'." >&2
        # Double-check if it exists now, in case of a race condition or non-fatal error reporting issue
        POST_CREATE_CHECK=$( $SUDO_CMD -u postgres PGPASSWORD= psql -X -tAc "SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}'" )
        if [ "$POST_CREATE_CHECK" != "1" ]; then
             exit 1 # Exit if creation truly failed
        else
             echo "WARN: User '${DB_USER}' seems to exist now despite reported error during creation." >&2
        fi
    fi
fi

# --- Optional: Grant privileges ---
# Example: Grant connect permission on a specific database
# read -p "Enter database name to grant CONNECT privilege to '${DB_USER}' (leave blank to skip): " GRANT_DB
# if [[ -n "$GRANT_DB" ]]; then
#    GRANT_SQL="GRANT CONNECT ON DATABASE \"${GRANT_DB}\" TO \"${DB_USER}\";"
#    echo "Granting CONNECT on database '${GRANT_DB}' to user '${DB_USER}'..."
#    if $SUDO_CMD -u postgres PGPASSWORD= psql -X -c "$GRANT_SQL"; then
#        echo "Privileges granted."
#    else
#        echo "WARN: Failed to grant privileges on '${GRANT_DB}'. Does the database exist?" >&2
#    fi
# fi

echo "--- PostgreSQL User Setup Script Finished ---"
echo "User '${DB_USER}' should now be configured."