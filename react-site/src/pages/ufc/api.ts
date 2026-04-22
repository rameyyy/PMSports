import type { Fight, Event } from './types';

export interface PaginationInfo {
  page: number;
  limit: number;
  total: number;
  total_pages: number;
}

export async function fetchUpcomingEvents(): Promise<Event[]> {
  const res = await fetch('/api/ufc/events/upcoming');
  if (!res.ok) throw new Error('Failed to fetch upcoming events');
  return res.json();
}

export async function fetchPastEvents(page = 1, limit = 10): Promise<{ events: Event[]; pagination: PaginationInfo }> {
  const res = await fetch(`/api/ufc/events/past?page=${page}&limit=${limit}`);
  if (!res.ok) throw new Error('Failed to fetch past events');
  return res.json();
}

export async function fetchEventFights(eventId: string): Promise<Fight[]> {
  const res = await fetch(`/api/ufc/events/${eventId}/fights`);
  if (!res.ok) throw new Error(`Failed to fetch fights for event ${eventId}`);
  return res.json();
}
