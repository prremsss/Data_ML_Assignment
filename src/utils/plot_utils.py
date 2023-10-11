import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

class PlotUtils:
    @staticmethod
    def plot_confusion_matrix(cm, classes, title, cmap=plt.cm.Blues):
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
        plt.tight_layout()
        plt.title(f'Confusion Matrix of {title}')
        plt.ylabel('True labels')
        plt.xlabel('Predicted labels')

    def plot_bar_chart(self,data):
        fig = px.bar(data, x='Jobs', y='Count',
                     color='Jobs',
                     color_discrete_sequence=px.colors.sequential.RdBu,
                     labels={'Jobs': 'Job', 'Count': 'Count'})

        fig.update_layout(
            title='Count of Jobs',
            xaxis_title='Job name',
            yaxis_title='Count',
            legend_title='Job names',
            showlegend=True,
            xaxis=dict(tickangle=-35),

            plot_bgcolor='black'
        )
        return fig

    def plot_pie(self,data):
        fig = px.pie(data, values='Count', names='Jobs',
                     title='Pie plot',
                     color_discrete_sequence=px.colors.sequential.RdBu,
                     hole=0,
                     labels={'Jobs': 'Job', 'Count': 'Count'},
                     opacity=0.8,
                     )

        # Customize the layout
        fig.update_layout(
            legend_title='Jobs',
            font=dict(family='Arial', size=16, color='black'),
            margin=dict(t=50, b=50, l=50, r=50)
        )
        return fig
