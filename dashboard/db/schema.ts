import { index, integer, sqliteTable, text, uniqueIndex } from 'drizzle-orm/sqlite-core'

export const users = sqliteTable('users', {
  id: text('id').primaryKey(),
  email: text('email').notNull(),
  role: text('role').notNull().default('member'),
  trialStartedAt: integer('trial_started_at', { mode: 'timestamp_ms' }),
  stripeCustomerId: text('stripe_customer_id'),
  stripeSubscriptionId: text('stripe_subscription_id'),
  subscriptionStatus: text('subscription_status').notNull().default('none'),
  subscriptionPeriodEnd: integer('subscription_period_end', { mode: 'timestamp_ms' }),
  createdAt: integer('created_at', { mode: 'timestamp_ms' }).notNull(),
  updatedAt: integer('updated_at', { mode: 'timestamp_ms' }).notNull(),
}, (table) => [
  uniqueIndex('users_email_unique').on(table.email),
  uniqueIndex('users_stripe_customer_unique').on(table.stripeCustomerId),
])

export const dailyPickAccess = sqliteTable('daily_pick_access', {
  id: text('id').primaryKey(),
  userId: text('user_id').notNull().references(() => users.id),
  accessDate: text('access_date').notNull(),
  targetDate: text('target_date').notNull(),
  stationId: text('station_id').notNull(),
  claimedAt: integer('claimed_at', { mode: 'timestamp_ms' }).notNull(),
}, (table) => [
  uniqueIndex('daily_pick_access_user_date_unique').on(table.userId, table.accessDate),
  index('daily_pick_access_station_date_index').on(table.stationId, table.targetDate),
])

export const stripeEvents = sqliteTable('stripe_events', {
  eventId: text('event_id').primaryKey(),
  eventType: text('event_type').notNull(),
  receivedAt: integer('received_at', { mode: 'timestamp_ms' }).notNull(),
})
