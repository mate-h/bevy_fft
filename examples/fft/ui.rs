//! Band-pass sliders and help text for the FFT demo.

use crate::band_pass::BandPassParams;
use bevy::{
    ecs::system::ParamSet,
    prelude::*,
    ui::{FocusPolicy, RelativeCursorPosition},
};
use bevy_fft::fft::FftSource;

pub const BRIGHTNESS_MIN: f32 = 0.25;
pub const BRIGHTNESS_MAX: f32 = 10.0;

/// Chain `Startup`: spawn the world in [`FftDemoStartup::Scene`], then band-pass UI in [`FftDemoStartup::BandPassUi`].
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum FftDemoStartup {
    Scene,
    BandPassUi,
}

#[derive(Component)]
pub struct BandPassHelpText;

#[derive(Component, Clone, Copy)]
pub enum BandPassSliderKind {
    BandPosition,
    BandWidth,
    OutputBrightness,
}

#[derive(Component)]
pub struct PositionSliderFill;

#[derive(Component)]
pub struct WidthSliderFill;

#[derive(Component)]
pub struct BrightnessSliderFill;

/// Top-left sliders and a bottom [`Text2d`] hint. Registered by [`BandPassUiPlugin`] during `Startup`.
fn setup_band_pass_ui(mut commands: Commands) {
    let label_style = TextFont {
        font_size: 15.0,
        ..default()
    };

    // Fixed label column so every slider track starts at the same x and has equal width.
    let label_col = 56.0;
    let track_w = 200.0;
    let row_gap_inner = 10.0;
    let row_w = label_col + row_gap_inner + track_w;
    let track_h = 22.0;
    let rail_color = Color::srgb(0.12, 0.12, 0.14);
    let fill_position = Color::srgb(0.25, 0.65, 0.95);
    let fill_width = Color::srgb(0.95, 0.55, 0.2);
    let fill_gain = Color::srgb(0.55, 0.9, 0.45);
    let panel_bg = Color::srgba(0.02, 0.02, 0.05, 0.88);

    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(10),
                top: px(10),
                padding: UiRect::all(px(12)),
                flex_direction: FlexDirection::Column,
                row_gap: px(10),
                ..default()
            },
            BackgroundColor(panel_bg),
        ))
        .with_children(|panel| {
            panel.spawn((
                Text::new("Band-pass"),
                TextFont {
                    font_size: 17.0,
                    ..default()
                },
                TextColor(Color::WHITE),
            ));

            panel
                .spawn((
                    Node {
                        width: px(row_w),
                        height: px(30),
                        flex_direction: FlexDirection::Row,
                        align_items: AlignItems::Center,
                        column_gap: px(row_gap_inner),
                        ..default()
                    },
                ))
                .with_children(|row| {
                    row.spawn((
                        Node {
                            width: px(label_col),
                            height: percent(100.0),
                            justify_content: JustifyContent::FlexEnd,
                            align_items: AlignItems::Center,
                            ..default()
                        },
                    ))
                    .with_children(|cell| {
                        cell.spawn((
                            Text::new("Center"),
                            label_style.clone(),
                            TextColor(Color::srgb(0.85, 0.85, 0.9)),
                        ));
                    });
                    row.spawn((
                        Node {
                            width: px(track_w),
                            height: px(track_h),
                            position_type: PositionType::Relative,
                            border_radius: BorderRadius::all(px(4)),
                            ..default()
                        },
                        BackgroundColor(rail_color),
                        Interaction::default(),
                        RelativeCursorPosition::default(),
                        FocusPolicy::Block,
                        BandPassSliderKind::BandPosition,
                    ))
                    .with_children(|track| {
                        track.spawn((
                            Node {
                                position_type: PositionType::Absolute,
                                left: px(0),
                                top: px(0),
                                bottom: px(0),
                                width: percent(0.0),
                                border_radius: BorderRadius::all(px(4)),
                                ..default()
                            },
                            BackgroundColor(fill_position),
                            Pickable::IGNORE,
                            PositionSliderFill,
                        ));
                    });
                });

            panel
                .spawn((
                    Node {
                        width: px(row_w),
                        height: px(30),
                        flex_direction: FlexDirection::Row,
                        align_items: AlignItems::Center,
                        column_gap: px(row_gap_inner),
                        ..default()
                    },
                ))
                .with_children(|row| {
                    row.spawn((
                        Node {
                            width: px(label_col),
                            height: percent(100.0),
                            justify_content: JustifyContent::FlexEnd,
                            align_items: AlignItems::Center,
                            ..default()
                        },
                    ))
                    .with_children(|cell| {
                        cell.spawn((
                            Text::new("Width"),
                            label_style.clone(),
                            TextColor(Color::srgb(0.85, 0.85, 0.9)),
                        ));
                    });
                    row.spawn((
                        Node {
                            width: px(track_w),
                            height: px(track_h),
                            position_type: PositionType::Relative,
                            border_radius: BorderRadius::all(px(4)),
                            ..default()
                        },
                        BackgroundColor(rail_color),
                        Interaction::default(),
                        RelativeCursorPosition::default(),
                        FocusPolicy::Block,
                        BandPassSliderKind::BandWidth,
                    ))
                    .with_children(|track| {
                        track.spawn((
                            Node {
                                position_type: PositionType::Absolute,
                                left: px(0),
                                top: px(0),
                                bottom: px(0),
                                width: percent(0.0),
                                border_radius: BorderRadius::all(px(4)),
                                ..default()
                            },
                            BackgroundColor(fill_width),
                            Pickable::IGNORE,
                            WidthSliderFill,
                        ));
                    });
                });

            panel
                .spawn((
                    Node {
                        width: px(row_w),
                        height: px(30),
                        flex_direction: FlexDirection::Row,
                        align_items: AlignItems::Center,
                        column_gap: px(row_gap_inner),
                        ..default()
                    },
                ))
                .with_children(|row| {
                    row.spawn((
                        Node {
                            width: px(label_col),
                            height: percent(100.0),
                            justify_content: JustifyContent::FlexEnd,
                            align_items: AlignItems::Center,
                            ..default()
                        },
                    ))
                    .with_children(|cell| {
                        cell.spawn((
                            Text::new("Gain"),
                            label_style.clone(),
                            TextColor(Color::srgb(0.85, 0.85, 0.9)),
                        ));
                    });
                    row.spawn((
                        Node {
                            width: px(track_w),
                            height: px(track_h),
                            position_type: PositionType::Relative,
                            border_radius: BorderRadius::all(px(4)),
                            ..default()
                        },
                        BackgroundColor(rail_color),
                        Interaction::default(),
                        RelativeCursorPosition::default(),
                        FocusPolicy::Block,
                        BandPassSliderKind::OutputBrightness,
                    ))
                    .with_children(|track| {
                        track.spawn((
                            Node {
                                position_type: PositionType::Absolute,
                                left: px(0),
                                top: px(0),
                                bottom: px(0),
                                width: percent(0.0),
                                border_radius: BorderRadius::all(px(4)),
                                ..default()
                            },
                            BackgroundColor(fill_gain),
                            Pickable::IGNORE,
                            BrightnessSliderFill,
                        ));
                    });
                });
        });

    commands.spawn((
        Text2d::default(),
        TextColor(Color::WHITE),
        TextFont {
            font_size: 15.0,
            ..default()
        },
        Transform::from_xyz(0.0, -420.0, 10.0),
        BandPassHelpText,
    ));
}

#[allow(clippy::type_complexity)]
pub fn band_pass_ui_sliders(
    mouse: Res<ButtonInput<MouseButton>>,
    mut queries: ParamSet<(
        Query<&mut BandPassParams>,
        Query<&mut FftSource>,
        Query<(
            &Interaction,
            &RelativeCursorPosition,
            &BandPassSliderKind,
        )>,
    )>,
) {
    let mut set_position: Option<f32> = None;
    let mut set_width: Option<f32> = None;
    let mut set_brightness: Option<f32> = None;

    for (interaction, rel, kind) in queries.p2().iter() {
        let active = matches!(*interaction, Interaction::Pressed)
            || (mouse.pressed(MouseButton::Left)
                && matches!(*interaction, Interaction::Hovered));
        if !active {
            continue;
        }
        let Some(n) = rel.normalized else {
            continue;
        };
        let t = (n.x + 0.5).clamp(0.0, 1.0);
        match kind {
            BandPassSliderKind::BandPosition => set_position = Some(t),
            BandPassSliderKind::BandWidth => set_width = Some(t),
            BandPassSliderKind::OutputBrightness => set_brightness = Some(t),
        }
    }

    {
        let mut q_params = queries.p0();
        let Ok(mut p) = q_params.single_mut() else {
            return;
        };
        if let Some(t) = set_position {
            p.band_center = t;
        }
        if let Some(t) = set_width {
            p.band_width = t;
        }
        p.band_center = p.band_center.clamp(0.0, 1.0);
        p.band_width = p.band_width.clamp(0.0, 1.0);
    }
    if let Some(t) = set_brightness {
        let mut q_src = queries.p1();
        if let Ok(mut src) = q_src.single_mut() {
            src.spatial_display_gain = BRIGHTNESS_MIN + t * (BRIGHTNESS_MAX - BRIGHTNESS_MIN);
        }
    }
}

#[allow(clippy::type_complexity)]
pub fn sync_band_pass_slider_fills(
    params: Query<&BandPassParams>,
    fft: Query<&FftSource>,
    mut nodes: ParamSet<(
        Query<&mut Node, With<PositionSliderFill>>,
        Query<&mut Node, With<WidthSliderFill>>,
        Query<&mut Node, With<BrightnessSliderFill>>,
    )>,
) {
    let Ok(p) = params.single() else {
        return;
    };

    let position_t = p.band_center.clamp(0.0, 1.0);
    let width_t = p.band_width.clamp(0.0, 1.0);
    if let Ok(mut node) = nodes.p0().single_mut() {
        node.width = percent(position_t * 100.0);
    }
    if let Ok(mut node) = nodes.p1().single_mut() {
        node.width = percent(width_t * 100.0);
    }
    if let Ok(src) = fft.single() {
        let g = src.spatial_display_gain;
        let gain_t =
            ((g - BRIGHTNESS_MIN) / (BRIGHTNESS_MAX - BRIGHTNESS_MIN)).clamp(0.0, 1.0);
        if let Ok(mut node) = nodes.p2().single_mut() {
            node.width = percent(gain_t * 100.0);
        }
    }
}

pub fn band_pass_help_text(
    params: Query<&BandPassParams>,
    fft: Query<&FftSource>,
    mut text_q: Query<&mut Text2d, With<BandPassHelpText>>,
) {
    let Ok(p) = params.single() else {
        return;
    };
    let Ok(mut text) = text_q.single_mut() else {
        return;
    };
    let gain = fft.single().map(|s| s.spatial_display_gain).unwrap_or(1.0);

    **text = format!(
        "Patterns 1-4. Center {:.2}. Width {:.2}. Gain {:.2}.",
        p.band_center, p.band_width, gain,
    );
}

pub struct BandPassUiPlugin;

impl Plugin for BandPassUiPlugin {
    fn build(&self, app: &mut App) {
        app.configure_sets(
            Startup,
            FftDemoStartup::BandPassUi.after(FftDemoStartup::Scene),
        )
        .add_systems(Startup, setup_band_pass_ui.in_set(FftDemoStartup::BandPassUi))
        .add_systems(
            Update,
            (
                band_pass_ui_sliders,
                sync_band_pass_slider_fills.after(band_pass_ui_sliders),
                band_pass_help_text.after(band_pass_ui_sliders),
            ),
        );
    }
}
