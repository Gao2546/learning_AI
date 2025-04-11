import 'express-session';

declare module 'express-session' {
  interface SessionData {
    user?: {
      id: number;
      username: string;
      current_chat_id?: number | string | null;
      [key: string]: any;
    };
  }
}

import { Session } from 'express-session';
import { IncomingMessage } from 'http';

declare module 'http' {
  interface IncomingMessage {
    session?: Session & Partial<SessionData>;
  }
}