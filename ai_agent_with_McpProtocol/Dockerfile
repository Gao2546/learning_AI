# Stage 1: Build the application
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files and install production dependencies
COPY package.json package-lock.json ./
RUN npm ci --only=production

# Copy the rest of the application code
COPY tsconfig.json ./
COPY src ./src
COPY public ./public

# Build the TypeScript code
RUN npm run build

# Stage 2: Create the production image
FROM node:20-alpine

WORKDIR /app

# Copy package.json (needed for npm start)
COPY package.json ./

# Copy dependencies from the builder stage
COPY --from=builder /app/node_modules ./node_modules

# Copy the built application from the builder stage
COPY --from=builder /app/build ./build

# Copy the public assets from the builder stage
COPY --from=builder /app/public ./public

# Expose the application port
EXPOSE 3000

# Command to run the application
CMD ["npm", "start"]