import pandas as pd
import os
import glob
from pathlib import Path

def combine_cricsheet_csv2(
    folder_path,
    output_matches,
    output_deliveries,
    league_name
):
    all_deliveries = []
    all_matches = []
    match_count = 0
    error_count = 0

    # Find all info files
    info_files = glob.glob(
        os.path.join(folder_path, '*_info.csv')
    )

    print(f"Found {len(info_files)} matches")
    print(f"Processing {league_name}...")

    for info_file in info_files:
        try:
            match_id = Path(info_file).stem\
                .replace('_info', '')

            delivery_file = info_file.replace(
                '_info.csv', '.csv'
            )

            if not os.path.exists(delivery_file):
                continue

            # ─────────────────────────────
            # Parse INFO file
            # 3 column format:
            # type, key, value
            # ─────────────────────────────
            info_rows = []
            with open(info_file, 'r',
                      encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(',', 2)
                    # Split max 2 times
                    # handles values with commas
                    if len(parts) >= 3:
                        info_rows.append({
                            'type':  parts[0],
                            'key':   parts[1],
                            'value': parts[2]
                        })
                    elif len(parts) == 2:
                        info_rows.append({
                            'type':  parts[0],
                            'key':   parts[1],
                            'value': ''
                        })

            info_df = pd.DataFrame(info_rows)

            # Only keep info rows
            info_df = info_df[
                info_df['type'] == 'info'
            ]

            # Helper to extract value by key
            def get_info(key):
                row = info_df[
                    info_df['key'] == key
                ]['value']
                return row.values[0].strip() \
                    if len(row) > 0 else None

            # Get both teams
            teams = info_df[
                info_df['key'] == 'team'
            ]['value'].tolist()
            teams = [t.strip() for t in teams]

            team1 = teams[0] \
                if len(teams) > 0 else None
            team2 = teams[1] \
                if len(teams) > 1 else None

            winner = get_info('winner')
            result = get_info('result')

            # Skip if no result
            if not winner and result != 'tie':
                continue

            match_info = {
                'id':            match_id,
                'season':        get_info('season'),
                'city':          get_info('city'),
                'venue':         get_info('venue'),
                'date':          get_info('date'),
                'team1':         team1,
                'team2':         team2,
                'toss_winner':   get_info(
                                   'toss_winner'),
                'toss_decision': get_info(
                                   'toss_decision'),
                'winner':        winner,
                'result':        result,
                'result_margin': get_info(
                                   'result_margin'),
                'method':        get_info('method'),
                'league':        league_name
            }

            all_matches.append(match_info)

            # ─────────────────────────────
            # Parse DELIVERY file
            # Already has correct format ✅
            # ─────────────────────────────
            try:
                del_df = pd.read_csv(
                    delivery_file,
                    low_memory=False
                )

                # Rename columns to match
                # IPL dataset format
                del_df = del_df.rename(columns={
                    'runs_off_bat': 'runs_batsman',
                    'innings':      'inning',
                    'start_date':   'date',
                    'noballs':      'noball_runs',
                    'wides':        'wide_runs',
                    'byes':         'bye_runs',
                    'legbyes':      'legbye_runs',
                    'extras':       'runs_extras',
                    'wicket_type':  'dismissal_kind',
                    'player_dismissed':
                                   'player_dismissed'
                })

                # Add total runs column
                del_df['runs_total'] = (
                    del_df['runs_batsman']\
                        .fillna(0) +
                    del_df['runs_extras']\
                        .fillna(0)
                )

                # Add league column
                del_df['league'] = league_name

                # Add winner for labeling
                del_df['match_winner'] = winner

                all_deliveries.append(del_df)
                match_count += 1

            except Exception as e:
                error_count += 1
                continue

        except Exception as e:
            error_count += 1
            continue

    print(f"Successfully processed: "
          f"{match_count} matches")
    print(f"Errors skipped: {error_count}")

    # ─────────────────────────────────────
    # Save outputs
    # ─────────────────────────────────────
    if all_matches:
        matches_df = pd.DataFrame(all_matches)
        os.makedirs(
            os.path.dirname(output_matches),
            exist_ok=True
        )
        matches_df.to_csv(
            output_matches, index=False
        )
        print(f"✅ Saved: {output_matches}")
        print(f"   Shape: {matches_df.shape}")
    else:
        print("❌ No matches to save!")

    if all_deliveries:
        deliveries_df = pd.concat(
            all_deliveries,
            ignore_index=True
        )
        deliveries_df.to_csv(
            output_deliveries, index=False
        )
        print(f"✅ Saved: {output_deliveries}")
        print(f"   Shape: {deliveries_df.shape}")
    else:
        print("❌ No deliveries to save!")

    return match_count


# ─────────────────────────────────────────
# RUN FOR T20 INTERNATIONALS
# ─────────────────────────────────────────
print("\n" + "="*50)
print("Processing T20 Internationals")
print("="*50)

combine_cricsheet_csv2(
    folder_path=r"C:\Users\srich\Downloads\t20s_male_csv2",
    output_matches="data/raw/t20wc_matches.csv",
    output_deliveries="data/raw/t20wc_deliveries.csv",
    league_name="T20I"
)

# ─────────────────────────────────────────
# RUN FOR BBL
# ─────────────────────────────────────────
print("\n" + "="*50)
print("Processing BBL")
print("="*50)

combine_cricsheet_csv2(
    folder_path=r"C:\Users\srich\Downloads\bbl_male_csv2",
    output_matches="data/raw/bbl_matches.csv",
    output_deliveries="data/raw/bbl_deliveries.csv",
    league_name="BBL"
)

print("\n" + "="*50)
print("✅ ALL DONE!")
print("="*50)
