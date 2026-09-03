CREATE TABLE `daily_pick_access` (
	`id` text PRIMARY KEY NOT NULL,
	`user_id` text NOT NULL,
	`access_date` text NOT NULL,
	`target_date` text NOT NULL,
	`station_id` text NOT NULL,
	`claimed_at` integer NOT NULL,
	FOREIGN KEY (`user_id`) REFERENCES `users`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE UNIQUE INDEX `daily_pick_access_user_date_unique` ON `daily_pick_access` (`user_id`,`access_date`);--> statement-breakpoint
CREATE INDEX `daily_pick_access_station_date_index` ON `daily_pick_access` (`station_id`,`target_date`);--> statement-breakpoint
CREATE TABLE `stripe_events` (
	`event_id` text PRIMARY KEY NOT NULL,
	`event_type` text NOT NULL,
	`received_at` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `users` (
	`id` text PRIMARY KEY NOT NULL,
	`email` text NOT NULL,
	`role` text DEFAULT 'member' NOT NULL,
	`trial_started_at` integer,
	`stripe_customer_id` text,
	`stripe_subscription_id` text,
	`subscription_status` text DEFAULT 'none' NOT NULL,
	`subscription_period_end` integer,
	`created_at` integer NOT NULL,
	`updated_at` integer NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX `users_email_unique` ON `users` (`email`);--> statement-breakpoint
CREATE UNIQUE INDEX `users_stripe_customer_unique` ON `users` (`stripe_customer_id`);