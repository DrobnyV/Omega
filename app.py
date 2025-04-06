import os
import re

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, redirect, url_for, request, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import requests
from datetime import datetime, timedelta
import random
import json
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
db = SQLAlchemy(app)

svm_model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')
team_stats_df = pd.read_csv('all_leagues_2022_2025.csv')

login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    favorite_teams = db.Column(db.String(500), default="")  # Comma-separated team names
    followers = db.relationship('User', secondary='follows',
                              primaryjoin='User.id==follows.c.follower_id',
                              secondaryjoin='User.id==follows.c.followed_id',
                              backref='following')
    shared_tickets = db.Column(db.Integer, default=0)

# Add follows table for follower relationships
follows = db.Table('follows',
    db.Column('follower_id', db.Integer, db.ForeignKey('user.id')),
    db.Column('followed_id', db.Integer, db.ForeignKey('user.id'))
)


class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    read = db.Column(db.Boolean, default=False)

    user = db.relationship('User', backref='notifications')

class Match(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    home_team = db.Column(db.String(100), nullable=False)
    away_team = db.Column(db.String(100), nullable=False)
    date = db.Column(db.String(100), nullable=False)
    league_id = db.Column(db.Integer, nullable=False)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)

class LastFetch(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    league_id = db.Column(db.Integer, unique=True, nullable=False)
    last_fetch_time = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

class MatchLike(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    match_id = db.Column(db.String(200), nullable=False)  # match_id bude string, protože je to unikátní ID zápasu
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref='match_likes')

class SharedTicket(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    ticket_data = db.Column(db.Text, nullable=False)  # JSON string of the ticket matches
    share_date = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref='shared_tickets_list')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

DEFAULT_TEAM_MAPPINGS = {
    "Stade Rennais FC 1901": "Rennes",
    "Paris Saint-Germain FC": "Paris S-G",
    "Club Atlético de Madrid": "Atlético Madrid",
    "RC Celta de Vigo": "Celta Vigo",
    "RCD Espanyol de Barcelona": "Espanyol",
    "FC Bayern München": "Bayern Munich",
    "Eintracht Frankfurt": "Eint Frankfurt",
    "Newcastle United FC": "Newcastle Utd",
    "Nottingham Forest FC": "Nott'ham Forest",
    "Wolverhampton Wanderers FC": "Wolves",
    "Manchester United FC": "Manchester Utd",
}


# Define team matching function
def find_team_match(match_team, stats_teams):
    if pd.isna(match_team):
        return None
    match_team_lower = match_team.lower().strip()
    for stats_team in stats_teams:
        stats_team_lower = stats_team.lower().strip()
        if (re.search(re.escape(stats_team_lower), match_team_lower) or
                re.search(re.escape(match_team_lower), stats_team_lower)):
            return stats_team
    return None


# Precompute team name mappings for efficiency
def generate_team_mappings():
    mappings = {}
    unmapped_teams = set()
    season = "2024-2025"  # Use only current season

    with app.app_context():  # Ensure DB queries run in app context
        for league in FREE_TIER_LEAGUES:
            league_id = league['id']
            league_name = league['name']
            # Get unique team names from stats for this league and season
            stats_teams = team_stats_df[
                (team_stats_df['League'] == league_name) &
                (team_stats_df['Season'] == season)
                ]['Squad'].unique()

            # Get unique team names from cached matches (if available)
            cached_matches = Match.query.filter_by(league_id=league_id).all()
            match_teams = set()
            for m in cached_matches:
                match_teams.add(m.home_team)
                match_teams.add(m.away_team)

            # Generate mapping for this league
            temp_mapping = {}
            for match_team in match_teams:
                # First check default mappings
                mapped_team = DEFAULT_TEAM_MAPPINGS.get(match_team)
                if mapped_team:
                    temp_mapping[match_team] = mapped_team
                else:
                    # Fall back to regex matching
                    matched_team = find_team_match(match_team, stats_teams)
                    if matched_team:
                        temp_mapping[match_team] = matched_team
                    else:
                        unmapped_teams.add(match_team)

            mappings[league_name] = temp_mapping

            if unmapped_teams:
                app.logger.warning(f"Unmapped teams in {league_name} {season}: {list(unmapped_teams)}")

    return mappings


# Define FREE_TIER_LEAGUES
FREE_TIER_LEAGUES = [
    {'id': 2021, 'name': 'Premier League'},
    {'id': 2002, 'name': 'Bundesliga'},
    {'id': 2019, 'name': 'Serie A'},
    {'id': 2014, 'name': 'La Liga'},
    {'id': 2015, 'name': 'Ligue 1'},
    # Add others as needed
]

# Initialize TEAM_MAPPINGS within app context at startup
with app.app_context():
    TEAM_MAPPINGS = generate_team_mappings()


def predict_winner(home_team, away_team, league_name=None):
    season = "2024-2025"  # Use only current season

    # Normalize team names using precomputed mappings
    if league_name and league_name in TEAM_MAPPINGS:
        home_team_mapped = TEAM_MAPPINGS[league_name].get(home_team, home_team)
        away_team_mapped = TEAM_MAPPINGS[league_name].get(away_team, away_team)
    else:
        home_team_mapped = DEFAULT_TEAM_MAPPINGS.get(home_team, home_team)
        away_team_mapped = DEFAULT_TEAM_MAPPINGS.get(away_team, away_team)

    # Fetch stats from team_stats_df for 2024-2025 only
    if league_name:
        home_stats = team_stats_df[
            (team_stats_df['Squad'] == home_team_mapped) &
            (team_stats_df['Season'] == season) &
            (team_stats_df['League'] == league_name)
            ]
        away_stats = team_stats_df[
            (team_stats_df['Squad'] == away_team_mapped) &
            (team_stats_df['Season'] == season) &
            (team_stats_df['League'] == league_name)
            ]
    else:
        home_stats = team_stats_df[
            (team_stats_df['Squad'] == home_team_mapped) &
            (team_stats_df['Season'] == season)
            ]
        away_stats = team_stats_df[
            (team_stats_df['Squad'] == away_team_mapped) &
            (team_stats_df['Season'] == season)
            ]

    if home_stats.empty or away_stats.empty:
        app.logger.warning(f"No stats for {home_team_mapped} or {away_team_mapped} in {season}")
        return "Unknown"

    home_stats = home_stats.iloc[0]
    away_stats = away_stats.iloc[0]

    # Construct feature vector
    features = np.array([[
        home_stats['W'], home_stats['D'], home_stats['L'], home_stats['Pts/MP'],
        home_stats['GD'], home_stats['xGD'],
        away_stats['W'], away_stats['D'], away_stats['L'], away_stats['Pts/MP'],
        away_stats['GD'], away_stats['xGD']
    ]])
    features_scaled = scaler.transform(features)
    prediction = svm_model.predict(features_scaled)[0]
    app.logger.debug(f"Predicting {home_team} vs {away_team}: {prediction}")
    return {0: "Draw", 1: home_team, 2: away_team}[prediction]

def fetch_all_matches():
    api_key = os.getenv('FOOTBALL_API_KEY')
    if not api_key:
        flash('API key not configured.', 'error')
        return {}

    matches_by_league = {}
    for league in FREE_TIER_LEAGUES:
        league_id = league['id']
        league_name = league['name']
        last_fetch = LastFetch.query.filter_by(league_id=league_id).first()
        cached_matches = Match.query.filter_by(league_id=league_id).all()

        if cached_matches and last_fetch and (datetime.utcnow() - last_fetch.last_fetch_time) < timedelta(days=1):
            matches = [{'home_team': m.home_team, 'away_team': m.away_team, 'date': m.date, 'league_id': m.league_id} for m in cached_matches]
        else:
            url = f'http://api.football-data.org/v4/competitions/{league_id}/matches?status=SCHEDULED'
            headers = {'X-Auth-Token': api_key}
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                matches_data = data.get('matches', [])
                db.session.query(Match).filter_by(league_id=league_id).delete()
                matches = []
                for match in matches_data[:10]:
                    home_team = match['homeTeam'].get('name')
                    away_team = match['awayTeam'].get('name')
                    if not home_team or not away_team:
                        continue
                    utc_date = match['utcDate']
                    date_obj = datetime.strptime(utc_date, '%Y-%m-%dT%H:%M:%SZ')
                    friendly_date = date_obj.strftime('%B %d, %Y, %I:%M %p UTC')
                    new_match = Match(home_team=home_team, away_team=away_team, date=friendly_date, league_id=league_id)
                    db.session.add(new_match)
                    matches.append({'home_team': home_team, 'away_team': away_team, 'date': friendly_date, 'league_id': league_id})
                if last_fetch:
                    last_fetch.last_fetch_time = datetime.utcnow()
                else:
                    db.session.add(LastFetch(league_id=league_id, last_fetch_time=datetime.utcnow()))
                db.session.commit()
            except requests.exceptions.RequestException as e:
                flash(f'Error fetching matches for league {league_id}: {str(e)}', 'error')
                matches = [{'home_team': m.home_team, 'away_team': m.away_team, 'date': m.date, 'league_id': m.league_id} for m in cached_matches] if cached_matches else []

        for match in matches:
            match['prediction'] = predict_winner(match['home_team'], match['away_team'], league_name)
            match['id'] = f"{match['league_id']}-{match['home_team']}-{match['away_team']}-{match['date']}"
            like_info = get_match_like_info(match['id'], current_user.id if current_user.is_authenticated else None)
            match['like_count'] = like_info['like_count']
            match['user_liked'] = like_info['user_liked']

        matches_by_league[str(league_id)] = matches

    return matches_by_league

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if len(username) < 3 or len(password) < 6:
            flash('Username must be 3+ characters and password 6+ characters.', 'error')
            return redirect(url_for('register'))
        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'error')
            return redirect(url_for('register'))
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('home'))
        flash('Login failed. Check your credentials.', 'error')
    return render_template('login.html')

@app.route('/', methods=['GET', 'POST'])
@login_required
def home():
    if 'selected_matches' not in session:
        session['selected_matches'] = []

    matches_by_league = fetch_all_matches()
    selected_league_id = request.form.get('league_id', '2021') if request.method == 'POST' else '2021'

    # Handle user search
    search_query = request.form.get('search_query', '').strip()
    searched_users = []
    if search_query:
        searched_users = User.query.filter(User.username.ilike(f'%{search_query}%')).filter(User.id != current_user.id).all()

    return render_template('home.html', username=current_user.username, matches_by_league=matches_by_league,
                           leagues=FREE_TIER_LEAGUES, selected_league_id=selected_league_id,
                           selected_matches=session['selected_matches'], searched_users=searched_users)

@app.route('/profile/<int:user_id>')
@login_required
def view_profile(user_id):
    user = User.query.get_or_404(user_id)
    shared_tickets = []
    for ticket in user.shared_tickets_list:
        try:
            ticket_data = json.loads(ticket.ticket_data)
            shared_tickets.append({
                'share_date': ticket.share_date,
                'ticket_data': ticket_data
            })
            logger.debug(f"Parsed ticket for {user.username}: {ticket_data}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse ticket_data: {ticket.ticket_data}")
            logger.error(f"Error: {str(e)}")
            shared_tickets.append({
                'share_date': ticket.share_date,
                'ticket_data': {'matches': [], 'ticket_date': 'Unknown', 'username': user.username}
            })

    return render_template('profile.html', username=user.username, user=user,
                           leagues=FREE_TIER_LEAGUES, shared_tickets=shared_tickets)

# Update the generate_ticket route to pass ticket data for sharing
@app.route('/generate_ticket', methods=['POST'])
@login_required
def generate_ticket():
    selected_matches_json = request.form.get('selected_matches', '[]')
    selected_matches = json.loads(selected_matches_json)
    if not selected_matches:
        flash('No matches selected for the ticket.', 'error')
        return redirect(url_for('home'))

    matches_by_league = fetch_all_matches()
    ticket_matches = []
    for league_id, matches in matches_by_league.items():
        for match in matches:
            if match['id'] in selected_matches:
                ticket_matches.append(match)

    if not ticket_matches:
        flash('Selected matches are no longer available.', 'error')
        return redirect(url_for('home'))

    ticket_date = datetime.utcnow().strftime('%B %d, %Y, %I:%M %p UTC')
    session['selected_matches'] = []
    session.modified = True
    # Removed notification creation here
    db.session.commit()

    ticket_data = {
        'matches': ticket_matches,
        'ticket_date': ticket_date,
        'username': current_user.username
    }
    ticket_data_json = json.dumps(ticket_data, ensure_ascii=False)

    return render_template('ticket.html', matches=ticket_matches, ticket_date=ticket_date,
                           username=current_user.username, leagues=FREE_TIER_LEAGUES,
                           ticket_data_json=ticket_data_json)  # Pass ticket_data_json to template
@app.route('/logout')
@login_required
def logout():
    session.pop('selected_matches', None)
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))



# Add this route to your existing app.py
@app.route('/profile')
@login_required
def profile():
    logger.debug(f"Accessing profile for user: {current_user.username}, ID: {current_user.id}")
    shared_tickets = []
    try:
        for ticket in current_user.shared_tickets_list:
            try:
                ticket_data = json.loads(ticket.ticket_data)
                shared_tickets.append({
                    'share_date': ticket.share_date,
                    'ticket_data': ticket_data
                })
                logger.debug(f"Parsed ticket: {ticket_data}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse ticket_data: {ticket.ticket_data}")
                logger.error(f"Error: {str(e)}")
                shared_tickets.append({
                    'share_date': ticket.share_date,
                    'ticket_data': {'matches': [], 'ticket_date': 'Unknown', 'username': current_user.username}
                })

        logger.debug(f"Shared tickets prepared for template: {shared_tickets}")
        return render_template('profile.html',
                               username=current_user.username,
                               user=current_user,  # Pass current_user as 'user' for consistency
                               leagues=FREE_TIER_LEAGUES,
                               shared_tickets=shared_tickets)
    except Exception as e:
        logger.error(f"Error in profile route: {str(e)}")
        flash('An error occurred while loading your profile.', 'error')
        return redirect(url_for('home'))

@app.route('/notifications')
@login_required
def get_notifications():
    notifications = Notification.query.filter_by(user_id=current_user.id).order_by(Notification.timestamp.desc()).limit(10).all()
    return jsonify([{
        'id': n.id,
        'message': n.message,
        'timestamp': n.timestamp.strftime('%Y-%m-%d %H:%M'),
        'read': n.read
    } for n in notifications])

@app.route('/notifications/read/<int:notif_id>', methods=['POST'])
@login_required
def mark_notification_read(notif_id):
    notification = Notification.query.get_or_404(notif_id)
    if notification.user_id == current_user.id:
        notification.read = True
        db.session.commit()  # Fixed the typo here
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Unauthorized'}), 403


@app.route('/notifications/read_all', methods=['POST'])
@login_required
def mark_all_notifications_read():
    try:
        # Fetch unread notifications for the current user
        notifications = Notification.query.filter_by(user_id=current_user.id, read=False).all()
        print(f"Found {len(notifications)} unread notifications for user {current_user.id}")

        # Mark each as read
        for notification in notifications:
            notification.read = True

        # Commit changes to the database
        db.session.commit()
        print("Database commit successful")

        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        print(f"Error marking notifications as read: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/like_match', methods=['POST'])
@login_required
def like_match():
    data = request.get_json()
    match_id = data.get('match_id')

    if not match_id:
        return jsonify({'success': False, 'error': 'No match ID provided'}), 400

    # Kontrola, zda uživatel již zápas lajkl
    existing_like = MatchLike.query.filter_by(user_id=current_user.id, match_id=match_id).first()

    if existing_like:
        # Pokud lajk existuje, odstraníme ho (toggle)
        db.session.delete(existing_like)
        db.session.commit()
        return jsonify({'success': True, 'liked': False})
    else:
        # Přidáme nový lajk
        new_like = MatchLike(user_id=current_user.id, match_id=match_id)
        db.session.add(new_like)
        db.session.commit()
        return jsonify({'success': True, 'liked': True})


# Přidejte funkci pro získání počtu lajků a stavu lajku pro konkrétní zápas
def get_match_like_info(match_id, user_id):
    like_count = MatchLike.query.filter_by(match_id=match_id).count()
    user_liked = MatchLike.query.filter_by(user_id=user_id, match_id=match_id).first() is not None
    return {'like_count': like_count, 'user_liked': user_liked}


# Modify the share_ticket route to notify followers
@app.route('/share_ticket', methods=['POST'])
@login_required
def share_ticket():
    ticket_data_json = request.form.get('ticket_data')
    if not ticket_data_json:
        flash('No ticket data provided.', 'error')
        return redirect(url_for('home'))

    logger.debug(f"Received ticket_data_json: {ticket_data_json}")

    # Increment shared_tickets counter
    current_user.shared_tickets += 1

    # Save the shared ticket
    shared_ticket = SharedTicket(
        user_id=current_user.id,
        ticket_data=ticket_data_json
    )
    db.session.add(shared_ticket)

    # Add notification for the user sharing the ticket
    ticket_data = json.loads(ticket_data_json)
    new_notification = Notification(
        user_id=current_user.id,
        message=f"You shared a ticket with {len(ticket_data['matches'])} matches!"
    )
    db.session.add(new_notification)

    # Notify followers
    for follower in current_user.followers:
        follower_notification = Notification(
            user_id=follower.id,
            message=f"{current_user.username} shared a ticket with {len(ticket_data['matches'])} matches!"
        )
        db.session.add(follower_notification)

    db.session.commit()
    return redirect(url_for('home'))


@app.route('/search_users', methods=['GET', 'POST'])
@login_required
def search_users():
    if request.method == 'POST':
        search_query = request.form.get('search_query', '').strip()
        if not search_query:
            flash('Please enter a username to search.', 'error')
            return redirect(url_for('search_users'))

        # Search for users (case-insensitive, partial match)
        users = User.query.filter(User.username.ilike(f'%{search_query}%')).filter(User.id != current_user.id).all()
        return render_template('search_users.html', users=users, leagues=FREE_TIER_LEAGUES)

    return render_template('search_users.html', users=[], leagues=FREE_TIER_LEAGUES)


# Add this route to follow/unfollow users
@app.route('/follow/<int:user_id>', methods=['POST'])
@login_required
def follow_user(user_id):
    user_to_follow = User.query.get_or_404(user_id)
    if user_to_follow == current_user:
        flash('You cannot follow yourself.', 'error')
        return redirect(url_for('home'))

    if user_to_follow in current_user.following:
        current_user.following.remove(user_to_follow)
        flash(f'You have unfollowed {user_to_follow.username}.', 'success')
    else:
        current_user.following.append(user_to_follow)
        flash(f'You are now following {user_to_follow.username}.', 'success')
        new_notification = Notification(
            user_id=user_to_follow.id,
            message=f"{current_user.username} started following you!"
        )
        db.session.add(new_notification)

    db.session.commit()
    return redirect(url_for('home'))


with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)