# Stage 1: Build the application
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files and install production dependencies
COPY ai_agent_with_McpProtocol/package.json ai_agent_with_McpProtocol/package-lock.json ./
RUN npm ci

# Copy the rest of the application code
COPY ai_agent_with_McpProtocol/tsconfig.json ./
COPY ai_agent_with_McpProtocol/src ./src
COPY ai_agent_with_McpProtocol/public ./public

# Build the TypeScript code
RUN npm run build

# Stage 2: Create the production image
FROM node:20-alpine

WORKDIR /app

RUN apk add --no-cache bash

# Copy package.json (needed for npm start)
COPY ai_agent_with_McpProtocol/package.json ./

# Copy dependencies from the builder stage
COPY --from=builder /app/node_modules ./node_modules

# Copy the built application from the builder stage
COPY --from=builder /app/build ./build

# setting_prompt.txt is copied with the rest of the build output above
COPY ai_agent_with_McpProtocol/build/setting_prompt.txt ./build/setting_prompt.txt

COPY ai_agent_with_McpProtocol/.env ./.env

# Copy the public assets from the builder stage
COPY --from=builder /app/public ./public

# Copy MCP server files
COPY mcp_BrowserBase ./mcp/mcp_BrowserBase

# Expose the application port
EXPOSE 3000

# Command to run the application
CMD ["npm", "start"]